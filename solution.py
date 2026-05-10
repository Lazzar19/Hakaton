# =========================================================
# ADVANCED METADRIVE AUTONOMOUS AGENT
# IMPROVED VERSION v3
# =========================================================

import os
import json
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# CONFIG
# =========================================================

LOG_DIR      = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_DIM    = 11
STACK_SIZE   = 4
INPUT_DIM    = STATE_DIM * STACK_SIZE  # 44
ACTION_DIM   = 9
REWARD_SCALE = 40.0

EPISODE_INDEX_PATH = os.path.join(LOG_DIR, "episode_index.txt")
PROGRESS_PATH = os.path.join(LOG_DIR, "progress.txt")


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
        if len(obs) > 5:
            return float(np.clip(float(obs[5]), -1.0, 1.0))
        return 0.0

    def extract_navi_info(self, obs):
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
        return np.nan_to_num(lidar, nan=0.0, posinf=1.0, neginf=0.0)

    def front_distance(self, lidar):
        sector = np.concatenate([lidar[-18:], lidar[:18]])
        return float(np.min(sector))

    def left_distance(self, lidar):
        return float(np.mean(lidar[25:70]))

    def right_distance(self, lidar):
        return float(np.mean(lidar[-70:-25]))

    def curve_ahead(self, lidar):
        front_left  = float(np.mean(lidar[10:40]))
        front_right = float(np.mean(lidar[-40:-10]))
        return float(np.clip(front_left - front_right, -1.0, 1.0))

    def lane_confidence(self, lane_offset):
        if np.isnan(lane_offset) or lane_offset < 0.0 or lane_offset > 1.0:
            return 0.0
        return 1.0


# =========================================================
# SENSOR HEALTH MONITOR
# =========================================================

class SensorHealthMonitor:

    def __init__(self):
        self.invalid_counter = 0

    def lidar_valid(self, lidar):
        if lidar is None or len(lidar) == 0 or np.any(np.isnan(lidar)) or np.mean(lidar) < 0.01:
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
            self.buffer.appendleft(np.zeros(self.state_dim, dtype=np.float32))
        return np.concatenate(list(self.buffer))


# =========================================================
# SMOOTH CONTROLLER
# =========================================================

class SmoothController:

    def __init__(self):
        self.prev_steer    = 0.0
        self.prev_throttle = 0.0

    def apply(self, steer, throttle):
        steer    = self.prev_steer    + 0.25 * (steer    - self.prev_steer)
        throttle = self.prev_throttle + 0.35 * (throttle - self.prev_throttle)
        self.prev_steer    = steer
        self.prev_throttle = throttle
        return steer, throttle


# =========================================================
# REWARD SYSTEM
# =========================================================

class RewardSystem:

    def compute(self, speed, lane_offset, steer, throttle,
                front_distance, crash, out_of_road,
                distance_delta, curve_strength, heading_error=0.0):

        reward = 0.0
        reward += distance_delta * 60.0
        reward += speed * 2.0
        reward -= abs(lane_offset - 0.5) * 1.5
        reward -= abs(heading_error) * 1.0
        reward -= abs(steer) * 0.05

        if throttle < -0.5:
            reward -= 0.15

        if abs(steer) > 0.3 and 0.2 < lane_offset < 0.8:
            reward += 0.5

        if curve_strength > 0.35 and speed > 0.6:
            reward -= 1.5

        if front_distance < 0.25:
            reward -= 1.5
        if front_distance < 0.15:
            reward -= 2.5

        reward += 0.01

        if crash:
            reward -= 40.0
        if out_of_road:
            reward -= 35.0

        return float(np.clip(reward / REWARD_SCALE, -1.0, 1.0))


# =========================================================
# DUELING DQN
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
# PRIORITIZED REPLAY BUFFER
# =========================================================

class PrioritizedReplayBuffer:
    def __init__(self, capacity=50000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.max_priority = 1.0  # Čuvamo maksimum ovde

    def push(self, state, action, reward, next_state, done):
        # Mnogo brže: Koristimo keširanu vrednost umesto max() petlje
        priority = self.max_priority
        if action in {1, 2, 4, 5, 7, 8}:
            priority *= 1.5

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = priority
            self.position = (self.position + 1) % self.capacity

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            priority = float(np.clip(abs(error) + 1e-5, 1e-5, 10.0))
            self.priorities[idx] = priority
            # Ažuriramo globalni max samo ako je novi prioritet veći
            self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities, dtype=np.float32)
        
        # FIX: Zamijeni sve NaN/inf sa minimalnim prioritetom
        priorities = np.where(np.isfinite(priorities), priorities, 1e-5)
        priorities = np.clip(priorities, 1e-5, None)
        
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # FIX: Finalna provjera prije choice-a
        if not np.isfinite(probabilities).all():
            probabilities = np.ones(len(self.buffer), dtype=np.float32)
            probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        total   = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        batch = list(zip(*samples))
        return batch, indices, torch.FloatTensor(weights).to(DEVICE)

    def __len__(self):
        return len(self.buffer)


# =========================================================
# DQN AGENT
# =========================================================

class DQNAgent:

    def __init__(self, state_dim=INPUT_DIM, action_dim=ACTION_DIM):
        self.action_dim    = action_dim
        self.policy_net    = DuelingDQN(state_dim, action_dim, dropout=0.3).to(DEVICE)
        self.target_net    = DuelingDQN(state_dim, action_dim, dropout=0.3).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer     = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.gamma         = 0.99
        self.epsilon       = 1.0
        self.epsilon_decay = 0.997
        self.epsilon_min   = 0.05
        self.tau           = 0.005

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q_values = self.policy_net(
                torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            )
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

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q       = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_q       = torch.clamp(next_q, -10, 10)
            target_q     = rewards + (1 - dones) * self.gamma * next_q

        td_errors = (target_q - current_q).detach().cpu().numpy()
        replay_buffer.update_priorities(indices, td_errors)

        loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.soft_update_target()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def soft_update_target(self):
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)


# =========================================================
# SOLUTION
# =========================================================

# =========================================================
# SOLUTION (MODIFIED) - full class with fixes inserted
# =========================================================
# (Assumes other classes like SensorProcessor, SensorHealthMonitor,
#  TemporalStacker, SmoothController, RewardSystem, DQNAgent,
#  PrioritizedReplayBuffer are defined elsewhere as in your project.)

class Solution:

    # --- Optional: temporarily less aggressive actions for safer testing ---
    ACTION_MAP = {
        0: (0.0,   0.4),
        1: (-0.2,  0.35),
        2: (0.2,   0.35),
        3: (0.0,  -0.5),
        4: (-0.4,  0.2),
        5: (0.4,   0.2),
        6: (0.0,   0.6),
        7: (-0.5, -0.1),
        8: (0.5,  -0.1)
    }

    def __init__(self, game):
        self._game         = game
        self.sensor        = SensorProcessor()
        self.monitor       = SensorHealthMonitor()
        self.smoother      = SmoothController()
        self.reward_system = RewardSystem()
        self.stacker       = TemporalStacker(state_dim=STATE_DIM, stack_size=STACK_SIZE)
        self.agent         = DQNAgent(state_dim=INPUT_DIM, action_dim=ACTION_DIM)
        self.replay_buffer = PrioritizedReplayBuffer(capacity=50000)

        self.prev_state        = None
        self.prev_action       = None
        self.prev_position     = None
        self.total_distance    = 0.0
        self.frame_index       = 0
        self.frames_since_turn = 0

        self.model_path    = os.path.join(LOG_DIR, "advanced_model.pt")
        self.episode_log   = []
        self.episode_index = self._load_episode_index()

        self._load_model()
        self._load_json_logs(pretrain=not os.path.exists(self.model_path))

        # Debug toggles
        self.debug_log_states = True   # set False to reduce stdout
        self.force_deterministic_eval = False  # set True to force epsilon=0 during evaluation

    @property
    def config(self):
        return {"image_observation": False}

    # -------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------

    def _load_episode_index(self):
        if os.path.exists(EPISODE_INDEX_PATH):
            try:
                with open(EPISODE_INDEX_PATH, "r") as f:
                    return int(f.read().strip())
            except:
                pass
        return 0

    def _save_episode_index(self):
        with open(EPISODE_INDEX_PATH, "w") as f:
            f.write(str(self.episode_index))

    @staticmethod
    def _continuous_to_discrete(steer, throttle):
        """Rekonstruiši diskretnu akciju iz kontinualnih vrijednosti."""
        return min(
            Solution.ACTION_MAP.keys(),
            key=lambda k: (
                (Solution.ACTION_MAP[k][0] - steer)    ** 2
                + (Solution.ACTION_MAP[k][1] - throttle) ** 2
            )
        )

    @staticmethod
    def _entry_to_state(e):
        """
        Statička metoda za konverziju log entry-a u state vektor.
        """
        return np.array([
            float(e.get("speed",          0.0)),
            float(e.get("lane_offset",    0.5)),
            float(e.get("front_distance", 1.0)),
            float(e.get("left_distance",  1.0)),
            float(e.get("right_distance", 1.0)),
            abs(float(e.get("lane_offset", 0.5)) - 0.5),
            1.0,
            float(e.get("asymmetry",      0.0)),
            float(e.get("heading_error",  0.0)),
            0.0,
            0.0,
        ], dtype=np.float32)

    # -------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------

    def do_iteration(self, simulator_output, user_input=None):
        self.frame_index += 1

        observation = simulator_output.get("observation")
        info        = simulator_output.get("info", {})

        if observation is None:
            return [0.0, 0.0]

        obs = np.asarray(observation, dtype=np.float32)

        speed         = self.sensor.extract_speed(obs)
        lane_offset   = self.sensor.extract_lane_offset(obs)
        heading_error = self.sensor.extract_heading_error(obs)
        navi          = self.sensor.extract_navi_info(obs)
        lidar         = self.sensor.extract_lidar(obs)
        position      = self.sensor.extract_position(obs)

        if lidar is None:
            return [0.0, -0.2]

        distance_delta = 0.0
        if self.prev_position is not None:
            distance_delta = float(np.linalg.norm(position - self.prev_position))
            self.total_distance += distance_delta
        self.prev_position = position

        if not self.monitor.lidar_valid(lidar):
            if getattr(self, "debug_log_states", False) and self.frame_index % 200 == 0:
                print(f"[LIDAR INVALID] frame={self.frame_index} invalid_counter={self.monitor.invalid_counter}")
            return [0.0, -0.3]

        front_distance  = self.sensor.front_distance(lidar)
        left_distance   = self.sensor.left_distance(lidar)
        right_distance  = self.sensor.right_distance(lidar)
        curve_strength  = abs(lane_offset - 0.5)
        lane_confidence = self.sensor.lane_confidence(lane_offset)
        asymmetry       = self.sensor.curve_ahead(lidar)

        # -----------------------------
        # NORMALIZATION (MATCH DATASET)
        # -----------------------------
        def clip01(x):
            return float(np.clip(x, 0.0, 1.0))

        def clip_m11(x):
            return float(np.clip(x, -1.0, 1.0))

        speed_norm      = clip01(speed / 30.0)
        lane_norm       = clip_m11(lane_offset)
        front_norm      = clip01(front_distance)
        left_norm       = clip01(left_distance)
        right_norm      = clip01(right_distance)
        curve_norm      = clip01(abs(lane_offset))
        asym_norm       = clip_m11(asymmetry)
        head_norm       = clip_m11(heading_error)
        navi0_norm      = clip_m11(navi[0])
        navi1_norm      = clip_m11(navi[1])

        state = np.array([
            speed_norm, lane_norm, front_norm,
            left_norm, right_norm,
            curve_norm, 1.0,
            asym_norm, head_norm,
            navi0_norm, navi1_norm,
        ], dtype=np.float32)

        if getattr(self, "debug_log_states", False) and self.frame_index % 500 == 0:
            print(f"[RAW->NORM] frame={self.frame_index} raw_speed={speed:.3f} speed_norm={speed_norm:.3f} "
                f"lane_raw={lane_offset:.3f} lane_norm={lane_norm:.3f} front_raw={front_distance:.3f} front_norm={front_norm:.3f}")

        # -----------------------------
        # STACKING
        # -----------------------------
        self.stacker.push(state)
        stacked_state = self.stacker.get()

        if getattr(self, "debug_log_states", False) and self.frame_index % 1000 == 0:
            print(f"[STACK CHECK] frame={self.frame_index} head={stacked_state[:11].tolist()} tail={stacked_state[-11:].tolist()}")

        # -----------------------------
        # ACTION SELECTION WITH SAFEGUARDS
        # -----------------------------
            # -----------------------------
    # ACTION SELECTION WITH STABILITY FIXES
    # -----------------------------
        turning_actions = {1, 2, 4, 5, 7, 8}

        # Compute Q-values (safe)
        q_vals = None
        try:
            with torch.no_grad():
                q_tensor = self.agent.policy_net(torch.FloatTensor(stacked_state).unsqueeze(0).to(DEVICE))
                q_vals = q_tensor.cpu().numpy().ravel()
        except Exception as e:
            # fallback to epsilon-greedy selection
            if getattr(self, "debug_log_states", False):
                print(f"[Q COMPUTE FAIL] {e}")
            action = self.agent.select_action(stacked_state)
            chosen = action
            q_vals = None

        # If q_vals computed, use tempered softmax; otherwise fallback to agent.select_action
        if q_vals is not None:
            if getattr(self, "debug_log_states", False) and self.frame_index % 200 == 0:
                q_mean = float(np.mean(q_vals)); q_std = float(np.std(q_vals))
                print(f"[Q STATS] frame={self.frame_index} mean={q_mean:.3f} std={q_std:.3f} q=" + ", ".join([f"{v:.3f}" for v in q_vals]))

            # Temperature softmax to smooth action preferences
            temp = 0.5  # niža temp = manje nasumičnosti; tune 0.4..0.8
            exp_q = np.exp((q_vals - q_vals.max()) / max(1e-6, temp))
            probs = exp_q / exp_q.sum()

            # Epsilon-greedy + tempered softmax
            if random.random() < max(0.0, self.agent.epsilon):
                chosen = random.randint(0, self.agent.action_dim - 1)
            else:
                chosen = int(np.random.choice(len(q_vals), p=probs))
        else:
            chosen = self.agent.select_action(stacked_state)

        # -----------------------------
        # ACTION HOLD / HYSTERESIS (min duration for switching)
        # -----------------------------
        # initialize counters if missing
        if not hasattr(self, "action_hold_count"):
            self.action_hold_count = 0
        if not hasattr(self, "action_hold_min"):
            self.action_hold_min = 4  # require action to be held for 4 frames before switching

        if self.prev_action is None:
            action = chosen
            self.action_hold_count = 1
        else:
            if chosen == self.prev_action:
                # continue holding
                self.action_hold_count = min(self.action_hold_min, self.action_hold_count + 1)
                action = chosen
            else:
                # if we haven't held previous action long enough, keep it
                if self.action_hold_count < self.action_hold_min:
                    action = self.prev_action
                    self.action_hold_count += 1
                else:
                    # allow switch but reset hold counter
                    action = chosen
                    self.action_hold_count = 1

        # update frames_since_turn
        if action in turning_actions:
            self.frames_since_turn = 0
        else:
            self.frames_since_turn += 1

        steer, throttle = self.ACTION_MAP[action]

        # -----------------------------
        # EMERGENCY OVERRIDE (SMOOTHED)
        # -----------------------------
        emergency_triggered = False
        # gentle braking when obstacle near
        if front_distance < 0.25:
            throttle = min(throttle, 0.0)
            emergency_triggered = True

        # ramped emergency braking when very close (use prev_throttle ramp)
        if front_distance < 0.15:
            target_thr = -1.0
            ramp_factor = 0.25
            prev_thr = getattr(self, "prev_throttle", 0.0)
            throttle = float(prev_thr + ramp_factor * (target_thr - prev_thr))
            # soften steer bias to avoid extreme instantaneous turns
            steer = 0.45 if left_distance > right_distance else -0.45
            emergency_triggered = True

        if emergency_triggered and getattr(self, "debug_log_states", False):
            print(f"[OVERRIDE] frame={self.frame_index} front={front_distance:.3f} left={left_distance:.3f} right={right_distance:.3f} steer={steer:.3f} thr={throttle:.3f}")

        if curve_strength > 0.25:
            throttle = min(throttle, 0.35)

        # -----------------------------
        # CONTINUOUS LOW-PASS + DEADZONE
        # -----------------------------
        # stronger low-pass to reduce jitter (alpha small -> slower change)
        alpha_steer = 0.12   # tune 0.08..0.2 (smaller = smoother)
        alpha_throt = 0.18

        prev_s = getattr(self, "prev_steer", 0.0)
        prev_t = getattr(self, "prev_throttle", 0.0)

        # apply low-pass on raw steer/throttle before tanh/scaling
        steer_lp = prev_s + alpha_steer * (steer - prev_s)
        throttle_lp = prev_t + alpha_throt * (throttle - prev_t)

        # apply existing scaling and nonlinearity
        steer_lp *= (1.0 - speed * 0.35)
        steer_lp = float(np.tanh(steer_lp * 1.15))

        # deadzone: ignore tiny steering/throttle oscillations
        steer_deadzone = 0.05
        throttle_deadzone = 0.03
        if abs(steer_lp) < steer_deadzone:
            steer_lp = 0.0
        if abs(throttle_lp) < throttle_deadzone:
            throttle_lp = 0.0
        else:
            throttle_lp = throttle_lp

        # final smoothing controller (keeps previous smoother but now receives filtered inputs)
        steer_final, throttle_final = self.smoother.apply(steer_lp, throttle_lp)

        # store prevs for next frame
        self.prev_steer = steer_final
        self.prev_throttle = throttle_final

        # -----------------------------
        # finalize and logging
        # -----------------------------
        steer = float(np.clip(steer_final, -1.0, 1.0))
        throttle = float(np.clip(throttle_final, -1.0, 1.0))

        crash       = info.get("crash", False)
        out_of_road = info.get("out_of_road", False)
        done        = (
            crash or out_of_road
            or simulator_output.get("terminated", False)
            or simulator_output.get("truncated",  False)
        )

        # compute steer delta for optional penalty
        steer_delta = steer - getattr(self, "last_logged_steer", 0.0)

        reward = self.reward_system.compute(
            speed=speed, lane_offset=lane_offset,
            steer=steer, throttle=throttle,
            front_distance=front_distance,
            crash=crash, out_of_road=out_of_road,
            distance_delta=distance_delta,
            curve_strength=curve_strength,
            heading_error=heading_error
        )

        # small local penalty for abrupt steering (keeps reward_system signature unchanged)
        reward = float(np.clip(reward - min(0.45, abs(steer_delta) * 0.45), -1.0, 1.0))

        # push transition
        if self.prev_state is not None:
            self.replay_buffer.push(
                self.prev_state, self.prev_action,
                reward, stacked_state, done
            )

        self.prev_state  = stacked_state
        self.prev_action = action
        self.last_logged_steer = steer

        # Debug: print Q-values occasionally
        if getattr(self, "debug_log_states", False) and self.frame_index % 200 == 0:
            try:
                with torch.no_grad():
                    q_check = self.agent.policy_net(torch.FloatTensor(stacked_state).unsqueeze(0).to(DEVICE))
                    qvals = q_check.cpu().numpy().ravel().tolist()
                    print(f"[Q VALS] frame={self.frame_index} q=" + ", ".join([f"{v:.3f}" for v in qvals]))
            except Exception as e:
                print(f"[Q LOG FAIL] {e}")

        # Train step
        self.agent.train_step(self.replay_buffer)

        # periodic diagnostics: replay distribution
        if getattr(self, "debug_log_states", False) and self.frame_index % 1000 == 0:
            try:
                from collections import Counter
                acts = [t[1] for t in self.replay_buffer.buffer]
                print("Replay action dist:", Counter(acts))
            except Exception:
                pass

        if self.frame_index % 1000 == 0:
            self._save_model()

        if self.frame_index % 100 == 0:
            print(
                f"[F{self.frame_index}] spd={speed:.2f} | lane={lane_offset:.2f} | "
                f"front={front_distance:.2f} | asym={asymmetry:.2f} | "
                f"head={heading_error:.2f} | eps={self.agent.epsilon:.3f} | act={action}"
            )

        # episode logging
        self.episode_log.append({
            "speed":           float(speed),
            "lane_offset":     float(lane_offset),
            "heading_error":   float(heading_error),
            "asymmetry":       float(asymmetry),
            "front_distance":  float(front_distance),
            "left_distance":   float(left_distance),
            "right_distance":  float(right_distance),
            "action":          int(action),
            "action_steering": float(steer),
            "action_throttle": float(throttle),
            "reward":          float(reward),
            "terminated":      bool(done)
        })

        if done or self.frame_index % 1000 == 0:
            self._save_episode()

        return [
            float(np.clip(steer,    -1.0, 1.0)),
            float(np.clip(throttle, -1.0, 1.0))
        ]


    # -------------------------------------------------------
    # SAVE / LOAD MODEL
    # -------------------------------------------------------

    def _save_model(self):
        try:
            torch.save(self.agent.policy_net.state_dict(), self.model_path)
            if self.debug_log_states:
                print(f"[MODEL SAVED] path={self.model_path}")
        except Exception as e:
            print(f"[MODEL SAVE FAILED] {e}")

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
                print(f"LOAD FAILED: {e} — brišem stari model.")
                try:
                    os.remove(self.model_path)
                except:
                    pass

    # -------------------------------------------------------
    # SAVE EPISODE
    # -------------------------------------------------------

    def _save_episode(self):
        if not self.episode_log:
            return

        path = os.path.join(LOG_DIR, f"episode_{self.episode_index}.json")
        with open(path, "w") as f:
            json.dump(self.episode_log, f)

        rewards  = [e["reward"]  for e in self.episode_log]
        speeds   = [e["speed"]   for e in self.episode_log]
        actions  = [e["action"]  for e in self.episode_log]
        frames   = len(self.episode_log)
        turned   = sum(1 for a in actions if a in {1, 2, 4, 5, 7, 8})
        terminal = self.episode_log[-1].get("terminated", False)
        cause    = "terminated" if terminal else "checkpoint"

        summary = (
            f"ep={self.episode_index:04d} | "
            f"frames={frames:5d} | "
            f"total_r={sum(rewards):+7.3f} | "
            f"avg_r={sum(rewards)/frames:+.4f} | "
            f"avg_spd={sum(speeds)/frames:.3f} | "
            f"turns={100*turned/frames:4.1f}% | "
            f"eps={self.agent.epsilon:.4f} | "
            f"buf={len(self.replay_buffer):5d} | "
            f"end={cause}"
        )

        with open(PROGRESS_PATH, "a") as f:
            f.write(summary + "\n")

        self.episode_log   = []
        self.episode_index += 1
        self._save_episode_index()

    # -------------------------------------------------------
    # OFFLINE TRENING IZ JSON LOGOVA
    # -------------------------------------------------------

    def _load_json_logs(self, pretrain=True):
        import time
        start_time = time.time()
        
        files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith(".json")], reverse=True)
        loaded = 0

        print(f"[OFFLINE] Učitavam logove iz {LOG_DIR}...")
        
        for file_name in files:
            if loaded >= self.replay_buffer.capacity: break
            
            try:
                path = os.path.join(LOG_DIR, file_name)
                with open(path, "r") as f:
                    episode = json.load(f)

                for i in range(len(episode) - 1):
                    curr = episode[i]
                    if "asymmetry" not in curr or "action" not in curr: continue

                    s = self._entry_to_state(curr)
                    ns = self._entry_to_state(episode[i+1])
                    
                    self.replay_buffer.push(
                        np.tile(s, STACK_SIZE), 
                        int(curr["action"]),
                        float(curr.get("reward", 0.0)),
                        np.tile(ns, STACK_SIZE),
                        bool(curr.get("terminated", False))
                    )
                    loaded += 1
            except Exception:
                continue

        print(f"[OFFLINE] Završeno za {time.time()-start_time:.2f}s. Učitano: {loaded}")

        if pretrain and loaded > 500:
            train_steps = 500 
            print(f"[OFFLINE TRAINING] {train_steps} koraka...")
            for _ in range(train_steps):
                self.agent.train_step(self.replay_buffer)
