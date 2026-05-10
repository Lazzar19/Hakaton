import os
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from collections import Counter

# --- KONFIGURACIJA ---
LOG_DIR     = "logs"
MODEL_PATH  = os.path.join(LOG_DIR, "advanced_model.pt")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE_DIM    = 11
STACK_SIZE   = 4
INPUT_DIM    = STATE_DIM * STACK_SIZE # 44
ACTION_DIM   = 9
VAL_SPLIT    = 0.2  # 20% fajlova ide u validaciju

# =========================================================
# MODEL: Dueling DQN sa regularizacijom
# =========================================================
class DuelingDQN(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, action_dim=ACTION_DIM, dropout=0.3):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout)
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
        f = self.feature(x)
        val = self.value_stream(f)
        adv = self.advantage_stream(f)
        return val + (adv - adv.mean(dim=1, keepdim=True))

# =========================================================
# DATASET: Sa podrškom za file-split
# =========================================================
class MetaDriveDataset(Dataset):
    def __init__(self, log_dir, file_list):
        self.samples = []
        
        def safe_f(val, default=0.0):
            try:
                res = float(val)
                return res if np.isfinite(res) else default
            except: return default

        for file in file_list:
            path = os.path.join(log_dir, file)
            with open(path, "r") as f:
                try: data = json.load(f)
                except: continue

            if len(data) < 5 or "asymmetry" not in data[0]:
                continue

            for i in range(STACK_SIZE - 1, len(data) - 1):
                def get_state(idx, _data=data):
                    d = _data[idx]
                    return [
                        np.clip(safe_f(d.get("speed")) / 30.0, 0, 1),
                        np.clip(safe_f(d.get("lane_offset")), -1, 1),
                        np.clip(safe_f(d.get("front_distance", 1.0)), 0, 1),
                        np.clip(safe_f(d.get("left_distance",  1.0)), 0, 1),
                        np.clip(safe_f(d.get("right_distance", 1.0)), 0, 1),
                        np.clip(abs(safe_f(d.get("lane_offset"))), 0, 1),
                        1.0,
                        np.clip(safe_f(d.get("asymmetry", 0.0)), -1, 1),
                        np.clip(safe_f(d.get("heading_error", 0.0)), -1, 1),
                        0.0, 0.0
                    ]

                s = np.array([v for j in range(i-(STACK_SIZE-1), i+1) for v in get_state(j)], dtype=np.float32)
                ns = np.array([v for j in range(i-(STACK_SIZE-2), i+2) for v in get_state(j)], dtype=np.float32)
                
                act = int(data[i].get("action", 0))
                rew = np.clip(safe_f(data[i].get("reward")), -1.0, 1.0)
                done = bool(data[i].get("terminated"))

                self.samples.append((s, act, rew, ns, done))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# =========================================================
# TRENING PROCES
# =========================================================
def train():
    # 1. SPLIT FAJLOVA (Srce borbe protiv overfittinga)
    all_files = [f for f in os.listdir(LOG_DIR) if f.endswith(".json")]
    if not all_files:
        print("Nema logova u folderu!")
        return
    
    random.shuffle(all_files)
    split_idx = int(len(all_files) * VAL_SPLIT)
    val_files = all_files[:split_idx]
    train_files = all_files[split_idx:]

    print(f"Učitavam {len(train_files)} fajlova za trening i {len(val_files)} za validaciju...")
    train_ds = MetaDriveDataset(LOG_DIR, train_files)
    val_ds   = MetaDriveDataset(LOG_DIR, val_files)

    # 2. SAMPLER ZA BALANS AKCIJA
    train_actions = [s[1] for s in train_ds.samples]
    counts = Counter(train_actions)
    weights = [1.0 / counts[a] for a in train_actions]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler)
    val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False)

    # 3. MODEL I OPTIMIZACIJA
    model = DuelingDQN(dropout=0.3).to(DEVICE)
    target_net = DuelingDQN().to(DEVICE)
    target_net.load_state_dict(model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.HuberLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 7

    print(f"Početak treninga na {DEVICE}...")

    for epoch in range(50):
        # --- TRAIN MODE ---
        model.train()
        train_loss = 0
        for s, a, r, ns, d in train_loader:
            s, a, r, ns, d = s.to(DEVICE).float(), a.to(DEVICE).long(), r.to(DEVICE).float(), ns.to(DEVICE).float(), d.to(DEVICE).float()
            
            # Augmentacija: Dodavanje blagog šuma na ulazne senzore (osim 1.0 markera)
            noise = (torch.randn_like(s) * 0.005).to(DEVICE)
            s = s + noise

            q_values = model(s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_actions = model(ns).argmax(1)
                max_next_q = target_net(ns).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target = r + (1 - d.float()) * 0.99 * max_next_q
            
            loss = criterion(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # --- VALIDATION MODE ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for s, a, r, ns, d in val_loader:
                s, a, r, ns, d = s.to(DEVICE).float(), a.to(DEVICE).long(), r.to(DEVICE).float(), ns.to(DEVICE).float(), d.to(DEVICE).float()
                q_v = model(s).gather(1, a.unsqueeze(1)).squeeze(1)
                n_a = model(ns).argmax(1)
                m_n_q = target_net(ns).gather(1, n_a.unsqueeze(1)).squeeze(1)
                t = r + (1 - d.float()) * 0.99 * m_n_q
                val_loss += criterion(q_v, t).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

                # --- after computing avg_val ---
        # smoothing + robust early stopping
        min_delta = 1e-4            # require at least this improvement
        ema_alpha = 0.25            # smoothing factor for val loss (0.0..1.0)
        if epoch == 0:
            smoothed_val = avg_val
        else:
            smoothed_val = ema_alpha * avg_val + (1 - ema_alpha) * smoothed_val

        # Use smoothed_val for early stopping decisions
        if smoothed_val + min_delta < best_val_loss:
            best_val_loss = smoothed_val
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            status = " [MODEL SAVED]"
        else:
            patience_counter += 1
            status = f" [Patience {patience_counter}/{max_patience}]"

        # Soft update target mreže
        for tp, mp in zip(target_net.parameters(), model.parameters()):
            tp.data.copy_(0.005 * mp.data + 0.995 * tp.data)

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f} | Smoothed Val: {smoothed_val:.6f}{status}")

        if patience_counter >= max_patience:
            print("Early stopping aktiviran.")
            break

if __name__ == "__main__":
    train()