# =========================================================
# ADVANCED METADRIVE AUTONOMOUS AGENT
# IMPROVED VERSION v2
# =========================================================
#
# PROMJENE U OVOJ VERZIJI:
#
# 1. State proširen sa 7 na 11 dimenzija:
#    - asymmetry   (LiDAR detekcija krivine)
#    - heading_error (nagib auta prema traci)
#    - navi[0], navi[1] (smjer sljedećeg checkpointa)
# 2. TemporalStacker: state_dim=11, stack_size=4 → input=44
# 3. DQNAgent: state_dim=44
# 4. DuelingDQN: input_dim=44
# 5. Soft target update (tau=0.005) umjesto hard copy
# 6. Forsirana eksploracija skretanja
# 7. Reward boost za uspješno skretanje
# 8. Reward normalizacija na [-1, 1]
#
# =========================================================

import os
import json
import math
import random
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# CONFIG
# =========================================================

LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

STATE_DIM   = 11
STACK_SIZE  = 4
INPUT_DIM   = STATE_DIM * STACK_SIZE  # 44
ACTION_DIM  = 9
REWARD_SCALE = 40.0  # normalizacioni faktor


# =========================================================
# SENSOR PROCESSOR
# =========================================================

class SensorProcessor:

    def extract_speed(self, obs):
        if len(obs) > 3:
            return float(obs[3])
        return 0.0

    def extract_lane_offset(self, obs):
        if len(obs) > 8:
            return float(obs[8])
        return 0.5

    def extract_heading_error(self, obs):
        # obs[5] je heading error prema centru trake u MetaDrive-u
        if len(obs) > 5:
            val = float(obs[5])
            return float(np.clip(val, -1.0, 1.0))
        return 0.0

    def extract_navi_info(self, obs):
        # obs[16:20] su vektori prema sljedećim navigacijskim checkpointima
        if len(obs) > 20:
            navi = obs[16:20].tolist()
            return [float(np.clip(v, -1.0, 1.0)) for v in navi]
        return [0.0, 0.0, 0.0, 0.0]

    def extract_position(self, obs):
        if len(obs) >= 2:
            return np.array([obs[0], obs[1]], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def extract_lidar(self, obs):
        if len(obs) < 240:
            return None
        lidar = obs[-240:]
        lidar = np.nan_to_num(lidar, nan=0.0, posinf=1.0, neginf=0.0)
        return lidar

    def front_distance(self, lidar):
        sector = np.concatenate([lidar[-18:], lidar[:18]])
        return float(np.min(sector))

    def left_distance(self, lidar):
        return float(np.mean(lidar[25:70]))

    def right_distance(self, lidar):
        return float(np.mean(lidar[-70:-25]))

    def curve_ahead(self, lidar):
        """
        Detektuje krivinu poređenjem lijeve i desne prednje četvrtine LiDAR-a.
        Pozitivna vrijednost = prepreka više sa lijeve strane = krivina desno.
        Negativna vrijednost = prepreka više sa desne strane = krivina lijevo.
        """
        front_left  = float(np.mean(lidar[10:40]))
        front_right = float(np.mean(lidar[-40:-10]))
        asymmetry   = float(np.clip(front_left - front_right, -1.0, 1.0))
        return asymmetry

    def lane_confidence(self, lane_offset):
        if np.isnan(lane_offset):
            return 0.0
        if lane_offset < 0.0 or lane_offset > 1.0:
            return 0.0
        return 1.0


# =========================================================
# SENSOR HEALTH MONITOR
# =========================================================

class SensorHealthMonitor:

    def __init__(self):
        self.invalid_counter = 0

    def lidar_valid(self, lidar):
        if lidar is None:
            self.invalid_counter += 1
            return False
        if len(lidar) == 0:
            self.invalid_counter += 1
            return False
        if np.any(np.isnan(lidar)):
            self.invalid_counter += 1
            return False
        if np.mean(lidar) < 0.01:
            self.invalid_counter += 1
            return False
        self.invalid_counter = max(0, self.invalid_counter - 1)
        return True

    def degraded(self):
        return self.invalid_counter > 5


# =========================================================
# TEMPORAL STACKER
# =========================================================

class TemporalStacker:

    def __init__(self, state_dim=STATE_DIM, stack_size=STACK_SIZE):
        self.state_dim  = state_dim
        self.stack_size = stack_size
        self.buffer     = deque(maxlen=stack_size)

    def reset(self):
        self.buffer.clear()

    def push(self, state):
        self.buffer.append(state)

    def get(self):
        while len(self.buffer) < self.stack_size:
            self.buffer.appendleft(
                np.zeros(self.state_dim, dtype=np.float32)
            )
        return np.concatenate(list(self.buffer))


# =========================================================
# SMOOTH CONTROLLER
# =========================================================

class SmoothController:

    def __init__(self):
        self.prev_steer    = 0.0
        self.prev_throttle = 0.0

    def apply(self, steer, throttle):
        # Povećani faktori za brže reakcije u krivinama
        steer    = self.prev_steer    + 0.25 * (steer    - self.prev_steer)
        throttle = self.prev_throttle + 0.35 * (throttle - self.prev_throttle)
        self.prev_steer    = steer
        self.prev_throttle = throttle
        return steer, throttle


# =========================================================
# REWARD SYSTEM
# =========================================================

class RewardSystem:

    def compute(
        self,
        speed,
        lane_offset,
        steer,
        throttle,
        front_distance,
        crash,
        out_of_road,
        distance_delta,
        curve_strength,
        heading_error=0.0
    ):
        reward = 0.0

        # Napredak
        reward += distance_delta * 35.0

        # Brzina
        reward += speed * 2.0

        # Držanje trake
        reward -= abs(lane_offset - 0.5) * 1.5

        # Heading error — kaznjavaj vožnju pod uglom prema traci
        reward -= abs(heading_error) * 1.0

        # Glatkoća
        reward -= abs(steer) * 0.05
        if throttle < -0.5:
            reward -= 0.15

        # Krivina — uspješno skretanje dok ostaje na traci
        if abs(steer) > 0.3 and 0.2 < lane_offset < 0.8:
            reward += 0.5

        # Kontrola brzine u krivini
        if curve_strength > 0.35 and speed > 0.6:
            reward -= 1.5

        # Opasnost od sudara
        if front_distance < 0.25:
            reward -= 1.5
        if front_distance < 0.15:
            reward -= 2.5

        # Preživljavanje — mali bonus svaki frejm
        reward += 0.01

        # Terminalni eventi
        if crash:
            reward -= 40.0
        if out_of_road:
            reward -= 35.0

        # Normalizacija na [-1, 1]
        return float(np.clip(reward / REWARD_SCALE, -1.0, 1.0))


# =========================================================
# DUELING DQN
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
        features   = self.feature(x)
        values     = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))


# =========================================================
# PRIORITIZED REPLAY BUFFER
# =========================================================

class PrioritizedReplayBuffer:

    def __init__(self, capacity=50000, alpha=0.6):
        self.capacity   = capacity
        self.alpha      = alpha
        self.buffer     = []
        self.priorities = []
        self.position   = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0

        # Boost prioriteta za rijetke akcije (skretanje)
        rare_actions = {1, 2, 4, 5, 7, 8}
        if action in rare_actions:
            max_priority *= 1.5

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position]     = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities    = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total   = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        batch = list(zip(*samples))
        return batch, indices, torch.FloatTensor(weights).to(DEVICE)

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            if np.isnan(error) or np.isinf(error):
                priority = 1e-5
            else:
                priority = abs(error) + 1e-5
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


# =========================================================
# DQN AGENT
# =========================================================

class DQNAgent:

    def __init__(self, state_dim=INPUT_DIM, action_dim=ACTION_DIM):
        self.action_dim = action_dim

        self.policy_net = DuelingDQN(state_dim, action_dim).to(DEVICE)
        self.target_net = DuelingDQN(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=1e-4
        )

        self.gamma         = 0.99
        self.epsilon       = 1.0
        self.epsilon_decay = 0.997
        self.epsilon_min   = 0.05
        self.tau           = 0.005  # soft update faktor

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values     = self.policy_net(state_tensor)
            return int(torch.argmax(q_values).item())

    def train_step(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < 500:
            return

        batch, indices, weights = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = batch

        states      = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions     = torch.LongTensor(actions).to(DEVICE)
        rewards     = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
        dones       = torch.FloatTensor(dones).to(DEVICE)

        current_q = self.policy_net(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q       = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            next_q    = torch.clamp(next_q, -10, 10)
            target_q  = rewards + (1 - dones) * self.gamma * next_q

        td_errors = (target_q - current_q).detach().cpu().numpy()
        replay_buffer.update_priorities(indices, td_errors)

        loss = (
            weights * F.smooth_l1_loss(current_q, target_q, reduction='none')
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Soft target update svaki korak
        self.soft_update_target()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def soft_update_target(self):
        for tp, pp in zip(
            self.target_net.parameters(),
            self.policy_net.parameters()
        ):
            tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)


# =========================================================
# SOLUTION
# =========================================================

class Solution:

    def __init__(self, game):
        self._game  = game

        self.sensor  = SensorProcessor()
        self.monitor = SensorHealthMonitor()
        self.smoother = SmoothController()
        self.reward_system = RewardSystem()

        self.stacker = TemporalStacker(
            state_dim=STATE_DIM,
            stack_size=STACK_SIZE
        )

        self.agent = DQNAgent(
            state_dim=INPUT_DIM,
            action_dim=ACTION_DIM
        )

        self.replay_buffer = PrioritizedReplayBuffer(capacity=50000)

        self.prev_state    = None
        self.prev_action   = None
        self.prev_position = None

        self.total_distance    = 0.0
        self.frame_index       = 0
        self.frames_since_turn = 0

        self.model_path   = os.path.join(LOG_DIR, "advanced_model.pt")
        self.episode_log  = []
        self.episode_index = 0

        self._load_model()
        self._load_json_logs()

    @property
    def config(self):
        return {"image_observation": False}

    # =====================================================
    # MAIN LOOP
    # =====================================================

    def do_iteration(self, simulator_output, user_input=None):
        self.frame_index += 1

        observation = simulator_output.get("observation")
        info        = simulator_output.get("info", {})

        if observation is None:
            return [0.0, 0.0]

        obs = np.asarray(observation, dtype=np.float32)

        # -----------------------------------------------------
        # EKSTRAKCIJA
        # -----------------------------------------------------
        speed         = self.sensor.extract_speed(obs)
        lane_offset   = self.sensor.extract_lane_offset(obs)
        heading_error = self.sensor.extract_heading_error(obs)
        navi          = self.sensor.extract_navi_info(obs)
        lidar         = self.sensor.extract_lidar(obs)
        position      = self.sensor.extract_position(obs)

        if lidar is None:
            return [0.0, -0.2]

        # -----------------------------------------------------
        # UDALJENOST
        # -----------------------------------------------------
        distance_delta = 0.0
        if self.prev_position is not None:
            distance_delta = float(np.linalg.norm(position - self.prev_position))
            self.total_distance += distance_delta
        self.prev_position = position

        # -----------------------------------------------------
        # VALIDACIJA SENZORA
        # -----------------------------------------------------
        if not self.monitor.lidar_valid(lidar):
            return [0.0, -0.3]

        # -----------------------------------------------------
        # LIDAR ANALIZA
        # -----------------------------------------------------
        front_distance = self.sensor.front_distance(lidar)
        left_distance  = self.sensor.left_distance(lidar)
        right_distance = self.sensor.right_distance(lidar)
        curve_strength = abs(lane_offset - 0.5)
        lane_confidence = self.sensor.lane_confidence(lane_offset)
        asymmetry      = self.sensor.curve_ahead(lidar)

        # -----------------------------------------------------
        # STATE (11 dimenzija)
        # -----------------------------------------------------
        state = np.array([
            speed,
            lane_offset,
            front_distance,
            left_distance,
            right_distance,
            curve_strength,
            lane_confidence,
            asymmetry,       # LiDAR detekcija krivine
            heading_error,   # nagib prema traci
            navi[0],         # smjer sljedećeg checkpointa
            navi[1],
        ], dtype=np.float32)

        self.stacker.push(state)
        stacked_state = self.stacker.get()

        # -----------------------------------------------------
        # AKCIJA + FORSIRANA EKSPLORACIJA SKRETANJA
        # -----------------------------------------------------
        self.frames_since_turn += 1
        turning_actions = {1, 2, 4, 5, 7, 8}

        # Svakih 200 frejmova bez skretanja forsiraj istraživanje
        if self.frames_since_turn > 200 and random.random() < 0.3:
            action = random.choice(list(turning_actions))
            self.frames_since_turn = 0
        else:
            action = self.agent.select_action(stacked_state)

        if action in turning_actions:
            self.frames_since_turn = 0

        steer, throttle = self._decode_action(action)

        # -----------------------------------------------------
        # EMERGENCY OVERRIDE
        # -----------------------------------------------------
        if front_distance < 0.25:
            throttle = min(throttle, 0.0)

        if front_distance < 0.15:
            throttle = -1.0
            steer    = 0.8 if left_distance > right_distance else -0.8

        # -----------------------------------------------------
        # KRIVINA — ogranič brzinu
        # -----------------------------------------------------
        if curve_strength > 0.25:
            throttle = min(throttle, 0.35)

        # -----------------------------------------------------
        # STABILIZACIJA
        # -----------------------------------------------------
        steer *= (1.0 - speed * 0.35)
        steer  = float(np.tanh(steer * 1.15))

        # -----------------------------------------------------
        # SMOOTHING
        # -----------------------------------------------------
        steer, throttle = self.smoother.apply(steer, throttle)

        # -----------------------------------------------------
        # TERMINACIJA
        # -----------------------------------------------------
        crash       = info.get("crash", False)
        out_of_road = info.get("out_of_road", False)
        done        = (
            crash
            or out_of_road
            or simulator_output.get("terminated", False)
            or simulator_output.get("truncated", False)
        )

        # -----------------------------------------------------
        # REWARD
        # -----------------------------------------------------
        reward = self.reward_system.compute(
            speed=speed,
            lane_offset=lane_offset,
            steer=steer,
            throttle=throttle,
            front_distance=front_distance,
            crash=crash,
            out_of_road=out_of_road,
            distance_delta=distance_delta,
            curve_strength=curve_strength,
            heading_error=heading_error
        )

        # -----------------------------------------------------
        # REPLAY BUFFER
        # -----------------------------------------------------
        if self.prev_state is not None:
            self.replay_buffer.push(
                self.prev_state,
                self.prev_action,
                reward,
                stacked_state,
                done
            )

        self.prev_state  = stacked_state
        self.prev_action = action

        # -----------------------------------------------------
        # TRENING
        # -----------------------------------------------------
        self.agent.train_step(self.replay_buffer)

        # -----------------------------------------------------
        # SNIMANJE
        # -----------------------------------------------------
        if self.frame_index % 1000 == 0:
            self._save_model()

        # -----------------------------------------------------
        # DEBUG ISPIS
        # -----------------------------------------------------
        if self.frame_index % 100 == 0:
            print(
                f"[F{self.frame_index}] "
                f"spd={speed:.2f} | lane={lane_offset:.2f} | "
                f"front={front_distance:.2f} | asym={asymmetry:.2f} | "
                f"head={heading_error:.2f} | "
                f"eps={self.agent.epsilon:.3f} | act={action}"
            )

        # -----------------------------------------------------
        # LOGOVANJE
        # -----------------------------------------------------
        self.episode_log.append({
            "speed":         float(speed),
            "lane_offset":   float(lane_offset),
            "heading_error": float(heading_error),
            "asymmetry":     float(asymmetry),
            "front_distance": float(front_distance),
            "left_distance": float(left_distance),
            "right_distance": float(right_distance),
            "action_steering": float(steer),
            "action_throttle": float(throttle),
            "reward":        float(reward),
            "terminated":    bool(done)
        })

        if done or self.frame_index % 1000 == 0:
            self._save_episode()

        return [
            float(np.clip(steer,    -1.0, 1.0)),
            float(np.clip(throttle, -1.0, 1.0))
        ]

    # =====================================================
    # ACTION DECODER
    # =====================================================

    def _decode_action(self, action):
        actions = {
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
        return actions.get(action, (0.0, 0.0))

    # =====================================================
    # SAVE / LOAD MODEL
    # =====================================================

    def _save_model(self):
        torch.save(self.agent.policy_net.state_dict(), self.model_path)

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.agent.epsilon = 0.3
            try:
                self.agent.policy_net.load_state_dict(
                    torch.load(self.model_path, map_location=DEVICE)
                )
                self.agent.soft_update_target()
                print("[MODEL LOADED]")
            except Exception as e:
                print("LOAD FAILED (vjerovatno stara dimenzija state-a):", e)
                print("Brišem stari model i počinjem ispočetka.")
                os.remove(self.model_path)

    # =====================================================
    # SAVE EPISODE
    # =====================================================

    def _save_episode(self):
        path = os.path.join(LOG_DIR, f"episode_{self.episode_index}.json")
        with open(path, "w") as f:
            json.dump(self.episode_log, f)
        self.episode_log   = []
        self.episode_index += 1

    # =====================================================
    # OFFLINE TRENING IZ JSON LOGOVA
    # =====================================================

    def _load_json_logs(self):
        files  = [f for f in os.listdir(LOG_DIR) if f.endswith(".json")]
        loaded = 0

        for file_name in files:
            try:
                path = os.path.join(LOG_DIR, file_name)
                with open(path, "r") as f:
                    episode = json.load(f)

                for i in range(len(episode) - 1):
                    current = episode[i]
                    nxt     = episode[i + 1]

                    # Provjeri da li log ima novi format (11 dimenzija)
                    # Stari logovi nemaju asymmetry i heading_error — preskoči ih
                    if "asymmetry" not in current:
                        continue

                    def entry_to_state(e):
                        return np.array([
                            float(e.get("speed", 0.0)),
                            float(e.get("lane_offset", 0.5)),
                            float(e.get("front_distance", 1.0)),
                            float(e.get("left_distance", 1.0)),
                            float(e.get("right_distance", 1.0)),
                            abs(float(e.get("lane_offset", 0.5)) - 0.5),
                            1.0,
                            float(e.get("asymmetry", 0.0)),
                            float(e.get("heading_error", 0.0)),
                            0.0,  # navi[0] — nije logovan u starim epizodama
                            0.0,  # navi[1]
                        ], dtype=np.float32)

                    # Offline logovi nemaju stacked state — koristimo single frame
                    # (manje idealno, ali bolje nego ništa)
                    s  = entry_to_state(current)
                    ns = entry_to_state(nxt)

                    # Stackujemo isti frame 4 puta za kompatibilnost sa INPUT_DIM=44
                    s_stacked  = np.tile(s,  STACK_SIZE)
                    ns_stacked = np.tile(ns, STACK_SIZE)

                    self.replay_buffer.push(
                        s_stacked,
                        int(current.get("action", 0)),
                        float(current.get("reward", 0.0)),
                        ns_stacked,
                        bool(current.get("terminated", False))
                    )
                    loaded += 1

            except Exception as e:
                print("JSON LOAD FAILED:", e)

        print(f"[OFFLINE DATA] {loaded} tranzicija učitano")

        if loaded > 5000:
            print("[OFFLINE TRAINING START]")
            for i in range(min(5000, loaded)):
                self.agent.train_step(self.replay_buffer)
                if i % 1000 == 0:
                    print(f"  offline step {i}/{min(5000, loaded)}")
            self.agent.soft_update_target()
            print("[OFFLINE TRAINING COMPLETE]")