import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from collections import Counter

LOG_DIR    = "logs"
MODEL_PATH = os.path.join(LOG_DIR, "advanced_model.pt")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mora biti identično sa solution.py
STATE_DIM    = 11
STACK_SIZE   = 4
INPUT_DIM    = STATE_DIM * STACK_SIZE  # 44
ACTION_DIM   = 9
REWARD_SCALE = 40.0


# =========================================================
# DUELING DQN — identična arhitektura kao u solution.py
# =========================================================

class DuelingDQN(nn.Module):

    def __init__(self, input_dim=INPUT_DIM, action_dim=ACTION_DIM):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        f   = self.feature(x)
        val = self.value_stream(f)
        adv = self.advantage_stream(f)
        return val + (adv - adv.mean(dim=1, keepdim=True))


# =========================================================
# DATASET
# =========================================================

class MetaDriveDataset(Dataset):

    def __init__(self, log_dir):
        self.samples = []

        self.action_map = {
            0: (0.0,   0.4),
            1: (-0.3,  0.4),
            2: (0.3,   0.4),
            3: (0.0,  -0.7),
            4: (-0.55, 0.2),
            5: (0.55,  0.2),
            6: (0.0,   0.75),
            7: (-0.8, -0.2),
            8: (0.8,  -0.2)
        }

        def safe_f(val, default=0.0):
            try:
                res = float(val)
                return res if np.isfinite(res) else default
            except:
                return default

        if not os.path.exists(log_dir):
            return

        skipped_old = 0

        for file in os.listdir(log_dir):
            if not file.endswith(".json"):
                continue

            with open(os.path.join(log_dir, file), "r") as f:
                try:
                    data = json.load(f)
                except:
                    continue

            if len(data) < 5:
                continue

            # Preskoči stare logove bez novih polja
            if "asymmetry" not in data[0]:
                skipped_old += 1
                continue

            for i in range(3, len(data) - 1):

                def get_state(idx):
                    d = data[idx]
                    return [
                        np.clip(safe_f(d.get("speed")) / 30.0, 0, 1),
                        np.clip(safe_f(d.get("lane_offset")), -1, 1),
                        np.clip(safe_f(d.get("front_distance", 1.0)), 0, 1),
                        np.clip(safe_f(d.get("left_distance", 1.0)), 0, 1),
                        np.clip(safe_f(d.get("right_distance", 1.0)), 0, 1),
                        np.clip(abs(safe_f(d.get("lane_offset"))), 0, 1),
                        1.0,
                        np.clip(safe_f(d.get("asymmetry", 0.0)), -1, 1),
                        np.clip(safe_f(d.get("heading_error", 0.0)), -1, 1),
                        0.0,  # navi[0]
                        0.0,  # navi[1]
                    ]

                # Temporal stack od 4 frejma
                s  = np.array(
                    [v for j in range(i - 3, i + 1) for v in get_state(j)],
                    dtype=np.float32
                )
                ns = np.array(
                    [v for j in range(i - 2, i + 2) for v in get_state(j)],
                    dtype=np.float32
                )

                # Akcija — nađi najbližu iz action_map
                l_s = safe_f(data[i].get("action_steering"))
                l_t = safe_f(data[i].get("action_throttle"))
                act = min(
                    self.action_map.keys(),
                    key=lambda k: (
                        (self.action_map[k][0] - l_s) ** 2
                        + (self.action_map[k][1] - l_t) ** 2
                    )
                )

                # Reward — već normalizovan u solution.py na [-1, 1]
                rew = np.clip(safe_f(data[i].get("reward")), -1.0, 1.0)

                self.samples.append((
                    s, act, rew, ns,
                    bool(data[i].get("terminated"))
                ))

        if skipped_old > 0:
            print(f"[INFO] Preskočeno {skipped_old} starih JSON fajlova (nema asymmetry polje)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# =========================================================
# TRENING
# =========================================================

def train():
    dataset = MetaDriveDataset(LOG_DIR)
    print(f"Učitano {len(dataset)} primjera.")

    if len(dataset) == 0:
        print("Nema podataka za trening. Pokreni agenta da skupi nove logove.")
        return

    # Distribucija akcija
    actions = [s[1] for s in dataset.samples]
    rewards = [s[2] for s in dataset.samples]
    print("Distribucija akcija:", Counter(actions))
    print(f"Avg reward: {sum(rewards)/len(rewards):.4f}")
    print(f"Pozitivnih: {sum(1 for r in rewards if r > 0)} | "
          f"Negativnih: {sum(1 for r in rewards if r < 0)}")

    # Provjeri i očisti korumpirani model
    if os.path.exists(MODEL_PATH):
        try:
            test_model = DuelingDQN().to(DEVICE)
            test_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            if torch.isnan(test_model.feature[0].weight).any():
                print("Stari model je korumpiran (NaN). Brišem ga.")
                os.remove(MODEL_PATH)
        except Exception as e:
            print(f"Model nije kompatibilan ({e}). Brišem ga.")
            os.remove(MODEL_PATH)

    # Model i target net
    model = DuelingDQN().to(DEVICE)
    target_model = DuelingDQN().to(DEVICE)

    def reset_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    model.apply(reset_weights)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Nastavak treninga od postojećeg modela.")

    target_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.HuberLoss()

    # WeightedRandomSampler za balansiranje akcija
    action_counts = Counter(actions)
    total         = len(dataset)
    sample_weights = [
        total / (len(action_counts) * action_counts[s[1]])
        for s in dataset.samples
    ]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=total, replacement=True
    )

    loader     = DataLoader(dataset, batch_size=64, sampler=sampler)
    num_batches = math.ceil(total / 64)
    tau         = 0.005  # soft update faktor

    model.train()

    for epoch in range(30):
        total_loss   = 0.0
        q_vals_list  = []
        target_list  = []
        reward_list  = []

        for s, a, r, ns, d in loader:
            s  = s.to(DEVICE).float()
            a  = a.to(DEVICE).long()
            r  = r.to(DEVICE).float()
            ns = ns.to(DEVICE).float()
            d  = d.to(DEVICE).float()

            q_vals = model(s).gather(1, a.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                # Double DQN: policy net bira akciju, target net procjenjuje
                next_actions = model(ns).argmax(1)
                max_next_q   = target_model(ns).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
                max_next_q = torch.clamp(max_next_q, -10, 10)

                # Gamma identičan sa solution.py (0.99)
                target = (r + (1.0 - d) * 0.99 * max_next_q).float()

            loss = criterion(q_vals, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Soft target update
            for tp, pp in zip(target_model.parameters(), model.parameters()):
                tp.data.copy_(tau * pp.data + (1.0 - tau) * tp.data)

            total_loss  += loss.item()
            q_vals_list.append(q_vals.mean().item())
            target_list.append(target.mean().item())
            reward_list.append(r.mean().item())

        avg_loss   = total_loss / num_batches
        avg_q      = sum(q_vals_list) / len(q_vals_list)
        avg_target = sum(target_list) / len(target_list)
        avg_reward = sum(reward_list) / len(reward_list)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Loss: {avg_loss:.6f} | "
            f"Avg Q: {avg_q:.4f} | "
            f"Avg Target: {avg_target:.4f} | "
            f"Avg Reward: {avg_reward:.4f}"
        )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Trening završen. Model snimljen u {MODEL_PATH}")


if __name__ == "__main__":
    train()