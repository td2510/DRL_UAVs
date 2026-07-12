"""
Gymnasium-compatible environment for the UAV trajectory optimization problem.

Implements the system model from the paper:
- Multiple UAVs with backscatter and caching
- Dynamic Time Splitting (DTS) for EH and data transmission
- Rotary-wing energy consumption model
- Shannon capacity-based throughput computation
"""
import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
import config as cfg


def compute_distance(a, b):
    """Euclidean distance between two 3D points."""
    return np.linalg.norm(a - b)


def compute_distance_sq(a, b):
    """Squared Euclidean distance between two 3D points."""
    return np.sum((a - b) ** 2)


def compute_e_fly(q_curr, q_next, dt=cfg.DELTA_T):
    """
    Compute propulsion energy for one time slot using the rotary-wing model.
    Eq. (9) in the paper.

    Args:
        q_curr: Current position (3,)
        q_next: Next position (3,)
        dt: Time slot duration

    Returns:
        E_fly: Energy consumed for propulsion (mW * s = mJ)
    """
    delta_sq = compute_distance_sq(q_curr, q_next)
    delta_val = math.sqrt(delta_sq)

    term1 = cfg.P_0 * (dt + cfg.K_1 / dt * delta_sq)

    inner = math.sqrt(dt**4 + cfg.K_2**2 * delta_sq**2) - cfg.K_2 * delta_sq
    # Clamp to avoid numerical issues with sqrt of negative
    inner = max(inner, 0.0)
    term2 = cfg.P_1 * math.sqrt(inner)

    term3 = cfg.K_3 * delta_val**3 / dt**2

    return term1 + term2 + term3


def compute_rate_uplink(d_su_sq, P_s):
    """
    Compute uplink rate R_u (source -> UAV). Eq. (21) in paper.
    Uses the same convention as GA/BCD code (no bandwidth multiplication).

    Args:
        d_su_sq: Squared distance from source to UAV
        P_s: BS transmit power

    Returns:
        R_u rate value (unitless, log2-based)
    """
    d_su_alpha = d_su_sq ** (cfg.ALPHA / 2)
    snr = cfg.THETA_0 * P_s / d_su_alpha
    return cfg.B_BANDWIDTH * math.log2(1 + snr)


def compute_rate_downlink(d_su_sq, d_du_sq, P_s, P_u, eta=cfg.ETA):
    """
    Compute downlink rate R_d (UAV -> destination). Eq. (22) in paper.
    Uses the same convention as GA/BCD code (no bandwidth multiplication).

    Args:
        d_su_sq: Squared distance from source to UAV
        d_du_sq: Squared distance from UAV to destination
        P_s: BS transmit power
        P_u: UAV transmit power
        eta: Backscatter coefficient

    Returns:
        R_d rate value (unitless, log2-based)
    """
    d_su_alpha = d_su_sq ** (cfg.ALPHA / 2)
    d_du_alpha = d_du_sq ** (cfg.ALPHA / 2)

    # P_u_bar = P_u * ceil(c_r), c_r = SIGMA_CACHE = 0.45 -> ceil(0.45) = 1
    ceil_cr = math.ceil(cfg.SIGMA_CACHE)
    P_u_bar = P_u * (1 + ceil_cr)  # Matches GA code: P_u * (1 + ceil(sigma))

    numerator = cfg.THETA * (eta * cfg.OMEGA_0 * P_s + P_u_bar * d_su_alpha)
    denominator = d_su_alpha * d_du_alpha

    return cfg.B_BANDWIDTH * math.log2(1 + numerator / denominator)


def compute_energy_harvested(d_su_sq, tau, dt=cfg.DELTA_T):
    """
    Compute harvested energy E_h for one time slot. Eq. (7) in paper.

    Args:
        d_su_sq: Squared distance from source to UAV
        tau: DTS ratio (fraction of slot for transmission)
        dt: Time slot duration

    Returns:
        E_h: Harvested energy (mJ)
    """
    d_su_alpha = d_su_sq ** (cfg.ALPHA / 2)
    return cfg.MU * (1 - tau) * dt * cfg.OMEGA_0 * cfg.P_WPT / d_su_alpha


class UAVEnvironment(gym.Env):
    """
    Multi-UAV trajectory optimization environment.

    The agent controls both UAVs simultaneously (centralized control).

    State (per UAV, concatenated for both):
        - Normalized position (x/X_MAX, y/Y_MAX, z/Z_MAX): 3 dims
        - Normalized remaining time: 1 dim
        - Distance to source (normalized): 1 dim
        - Distance to destination (normalized): 1 dim
        - Cumulative energy balance (harvested - consumed, normalized): 1 dim
        - Cumulative throughput (normalized): 1 dim

    Total state: 8 * 2 UAVs = 16 dims

    Action (per UAV, concatenated):
        - Velocity direction (dx, dy, dz) in [-1, 1]: 3 dims
        - DTS ratio tau in [0, 1]: 1 dim
        - P_u normalized in [0, 1]: 1 dim
        - P_s normalized in [0, 1]: 1 dim

    Total action: 6 * 2 UAVs = 12 dims
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.num_uavs = cfg.NUM_UAVS
        self.n_slots = cfg.N_SLOTS
        self.dt = cfg.DELTA_T

        # State and action dimensions
        self.state_dim_per_uav = 15
        self.action_dim_per_uav = 6
        self.state_dim = self.state_dim_per_uav * self.num_uavs
        self.action_dim = self.action_dim_per_uav * self.num_uavs

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.action_dim,), dtype=np.float32
        )

        # UAV start/end positions
        self.q_init = [cfg.Q_I1.copy(), cfg.Q_I2.copy()]
        self.q_final = [cfg.Q_F1.copy(), cfg.Q_F2.copy()]

        self.render_mode = render_mode
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0

        # Current positions
        self.positions = [q.copy() for q in self.q_init]

        # Tracking variables per UAV
        self.cum_throughput = [0.0] * self.num_uavs      # sum of tau*dt*R_d
        self.cum_R_u = [0.0] * self.num_uavs             # sum of tau*dt*R_u
        self.cum_R_d = [0.0] * self.num_uavs             # sum of tau*dt*R_d
        self.cum_energy_consumed = [0.0] * self.num_uavs  # sum of E_fly + tau*dt*(P_b + P_u)
        self.cum_energy_harvested = [0.0] * self.num_uavs # sum of E_h
        self.velocities = [np.zeros(3) for _ in range(self.num_uavs)] # current velocity

        # History for plotting
        self.trajectory_history = [[q.copy()] for q in self.q_init]
        self.reward_history = []
        self.throughput_per_step = [[] for _ in range(self.num_uavs)]

        state = self._get_state()
        return state, {}

    def _get_state(self):
        """Construct the observation vector."""
        states = []
        for m in range(self.num_uavs):
            pos = self.positions[m]

            # Normalize values for better learning
            norm_pos = np.array([
                pos[0] / cfg.X_MAX,
                pos[1] / cfg.Y_MAX,
                pos[2] / cfg.Z_MAX
            ])
            norm_time = (self.n_slots - self.step_count) / self.n_slots
            
            # Relative vectors give directional gradients to the agent
            rel_w_s = (cfg.W_S - pos) / 30.0
            rel_w_d = (cfg.W_D - pos) / 30.0
            rel_q_f = (self.q_final[m] - pos) / 30.0
            
            energy_balance = (self.cum_energy_harvested[m] - self.cum_energy_consumed[m])
            norm_energy = np.tanh(energy_balance / 1000.0)  # Soft normalization
            norm_throughput = self.cum_throughput[m] / max(cfg.S_DEMAND, 1.0)

            uav_state = np.array([
                norm_pos[0], norm_pos[1], norm_pos[2],
                norm_time,
                rel_w_s[0], rel_w_s[1], rel_w_s[2],
                rel_w_d[0], rel_w_d[1], rel_w_d[2],
                rel_q_f[0], rel_q_f[1], rel_q_f[2],
                norm_energy,
                norm_throughput
            ], dtype=np.float32)
            states.append(uav_state)

        return np.concatenate(states)

    def step(self, action):
        """
        Execute one time step.

        Args:
            action: np.array of shape (12,) in [-1, 1]
                Per UAV (6 dims each):
                [dx, dy, dz, tau, P_u_norm, P_s_norm]

        Returns:
            state, reward, terminated, truncated, info
        """
        reward = 0.0
        info = {"throughput": 0.0, "energy_violation": 0.0, "constraints_ok": True}

        total_step_throughput = 0.0

        for m in range(self.num_uavs):
            offset = m * self.action_dim_per_uav
            a = action[offset:offset + self.action_dim_per_uav]

            # Parse actions
            accel_dir = a[0:3]  # in [-1, 1]
            tau = (a[3] + 1.0) / 2.0  # Map [-1,1] -> [0,1]
            tau = np.clip(tau, 0.01, 0.99)  # Avoid extreme values
            P_u = (a[4] + 1.0) / 2.0 * cfg.P_U_MAX  # Map [-1,1] -> [0, P_U_MAX]
            P_s = (a[5] + 1.0) / 2.0 * cfg.P_S_MAX  # Map [-1,1] -> [0, P_S_MAX]

            # Landing override: force UAV toward final position in last N steps
            remaining = self.n_slots - self.step_count
            if remaining <= cfg.LANDING_STEPS:
                # Fly straight toward destination at controlled speed
                to_final = self.q_final[m] - self.positions[m]
                dist_final = np.linalg.norm(to_final)
                if dist_final > 0.01:
                    desired_speed = min(dist_final / (remaining * self.dt + 1e-6), cfg.V_MAX)
                    new_vel = (to_final / dist_final) * desired_speed
                else:
                    new_vel = np.zeros(3)
                displacement = new_vel * self.dt
                new_pos = self.positions[m] + displacement
                self.velocities[m] = new_vel
            else:
                # Normal acceleration-based movement
                accel = accel_dir * cfg.A_MAX
                new_vel = self.velocities[m] + accel * self.dt
                
                # Enforce max speed limit
                speed = np.linalg.norm(new_vel)
                if speed > cfg.V_MAX:
                    new_vel = (new_vel / speed) * cfg.V_MAX

                # Compute new position
                displacement = new_vel * self.dt
                new_pos = self.positions[m] + displacement
                self.velocities[m] = new_vel

            # Enforce spatial bounds
            new_pos[0] = np.clip(new_pos[0], 0.0, cfg.X_MAX)
            new_pos[1] = np.clip(new_pos[1], 0.0, cfg.Y_MAX)
            new_pos[2] = np.clip(new_pos[2], cfg.Z_MIN, cfg.Z_MAX)

            # Enforce speed constraint (already handled by clipping vel_norm)
            actual_disp = np.linalg.norm(new_pos - self.positions[m])
            if actual_disp > cfg.DELTA_D + 1e-6:
                # Should not happen due to clipping, but safety check
                direction = (new_pos - self.positions[m]) / actual_disp
                new_pos = self.positions[m] + direction * cfg.DELTA_D

            # Compute distances
            d_su_sq = compute_distance_sq(new_pos, cfg.W_S)
            d_du_sq = compute_distance_sq(new_pos, cfg.W_D)

            # Avoid division by zero
            d_su_sq = max(d_su_sq, 1e-6)
            d_du_sq = max(d_du_sq, 1e-6)

            # Energy computations (compute E_fly first to check energy budget)
            E_fly = compute_e_fly(self.positions[m], new_pos)

            # Energy-aware tau clamping: if energy is already depleted,
            # reduce tau to force more energy harvesting time
            energy_budget = self.cum_energy_harvested[m] - self.cum_energy_consumed[m]
            if energy_budget < E_fly:
                # Not enough energy even for flying, force maximum EH
                tau = max(0.01, min(tau, 0.1))

            # Compute rates
            R_u = compute_rate_uplink(d_su_sq, P_s)
            R_d = compute_rate_downlink(d_su_sq, d_du_sq, P_s, P_u)

            # Throughput contribution this slot
            throughput_slot = tau * self.dt * R_d

            E_consumed_slot = E_fly + tau * self.dt * (cfg.P_B + P_u)
            E_harvested_slot = compute_energy_harvested(d_su_sq, tau)

            # Update cumulative values
            self.cum_throughput[m] += throughput_slot
            self.cum_R_u[m] += tau * self.dt * R_u
            self.cum_R_d[m] += throughput_slot
            self.cum_energy_consumed[m] += E_consumed_slot
            self.cum_energy_harvested[m] += E_harvested_slot

            # Update position
            self.positions[m] = new_pos
            self.trajectory_history[m].append(new_pos.copy())
            self.throughput_per_step[m].append(throughput_slot)

            # Step reward: throughput contribution
            reward += cfg.THROUGHPUT_SCALE * throughput_slot
            total_step_throughput += throughput_slot

            # Dense reward to guide UAV closer to Source and Destination
            dist_S = math.sqrt(d_su_sq)
            dist_D = math.sqrt(d_du_sq)
            # Bonus increases when distance is smaller
            proximity_bonus = cfg.LAMBDA_PROXIMITY * (1.0 / (dist_S + 1.0) + 1.0 / (dist_D + 1.0))
            reward += proximity_bonus

            # Midpoint bonus: time-weighted to create V-shape (not U-shape)
            # Gaussian peak at mid-flight forces UAV to reach midpoint at the middle,
            # then ascend in the second half
            remaining = self.n_slots - self.step_count
            if remaining > cfg.LANDING_STEPS:
                t_norm = self.step_count / self.n_slots  # 0 to 1
                time_weight = math.exp(-8.0 * (t_norm - 0.5) ** 2)  # Peak at mid-flight

                dist_mid = compute_distance(new_pos, cfg.MIDPOINT)
                midpoint_bonus = cfg.LAMBDA_MIDPOINT * time_weight * (1.0 / (dist_mid + 1.0))
                reward += midpoint_bonus

                # Loitering bonus: reward being very close to midpoint (not just low speed)
                if dist_mid < cfg.LOITER_RADIUS:
                    closeness = 1.0 - dist_mid / cfg.LOITER_RADIUS
                    loiter_bonus = cfg.LAMBDA_LOITER * time_weight * closeness
                    reward += loiter_bonus

            # Energy constraint penalty (causal, per slot)
            energy_violation = max(0.0, self.cum_energy_consumed[m] - self.cum_energy_harvested[m])
            if energy_violation > 0:
                reward -= cfg.LAMBDA_ENERGY * energy_violation / 100.0
                info["energy_violation"] += energy_violation
                info["constraints_ok"] = False

        self.step_count += 1
        terminated = (self.step_count >= self.n_slots)
        truncated = False

        # Terminal rewards/penalties
        if terminated:
            for m in range(self.num_uavs):
                # Penalty: data relay constraint violation
                relay_violation = max(0.0, self.cum_R_d[m] - self.cum_R_u[m] - cfg.SIGMA_CACHE * cfg.S_DEMAND)
                if relay_violation > 0:
                    reward -= cfg.LAMBDA_DATA_RELAY * relay_violation
                    info["constraints_ok"] = False

                # Penalty: demand not met
                demand_deficit = max(0.0, cfg.S_DEMAND - self.cum_R_d[m])
                if demand_deficit > 0:
                    reward -= cfg.LAMBDA_DEMAND * demand_deficit
                    info["constraints_ok"] = False

                # Penalty: not reaching final position (quadratic for stronger pull)
                dist_to_final = compute_distance(self.positions[m], self.q_final[m])
                if dist_to_final > cfg.DELTA_D:
                    reward -= cfg.LAMBDA_BOUNDARY * dist_to_final ** 2

            # Bonus if all constraints satisfied
            total_throughput = sum(self.cum_throughput)
            if info["constraints_ok"] and total_throughput >= cfg.S_DEMAND:
                reward += 5.0  # Bonus for feasible solution

        info["throughput"] = total_step_throughput
        info["total_throughput"] = sum(self.cum_throughput)

        state = self._get_state()
        self.reward_history.append(reward)

        return state, reward, terminated, truncated, info

    def get_results(self):
        """Return summary of the episode results."""
        return {
            "total_throughput": sum(self.cum_throughput),
            "throughput_per_uav": list(self.cum_throughput),
            "cum_R_u": list(self.cum_R_u),
            "cum_R_d": list(self.cum_R_d),
            "energy_consumed": list(self.cum_energy_consumed),
            "energy_harvested": list(self.cum_energy_harvested),
            "energy_feasible": all(
                self.cum_energy_harvested[m] >= self.cum_energy_consumed[m]
                for m in range(self.num_uavs)
            ),
            "relay_feasible": all(
                self.cum_R_u[m] + cfg.SIGMA_CACHE * cfg.S_DEMAND >= self.cum_R_d[m]
                for m in range(self.num_uavs)
            ),
            "demand_met": all(
                self.cum_R_d[m] >= cfg.S_DEMAND
                for m in range(self.num_uavs)
            ),
            "final_positions": [pos.copy() for pos in self.positions],
            "trajectories": [
                np.array(traj) for traj in self.trajectory_history
            ],
        }
