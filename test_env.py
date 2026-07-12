"""
Unit tests for verifying the UAV environment physics match the GA code.

Validates:
- Rate formulas (R_u, R_d) against GA's fitness function
- Energy model (E_fly) against GA's energy computation
- Energy harvesting formula
- Environment step mechanics
"""
import sys
import os
import numpy as np
import math

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from uav_env import (
    compute_distance,
    compute_distance_sq,
    compute_e_fly,
    compute_rate_uplink,
    compute_rate_downlink,
    compute_energy_harvested,
    UAVEnvironment,
)


def test_distance():
    """Test distance computation."""
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([3.0, 4.0, 0.0])
    assert abs(compute_distance(a, b) - 5.0) < 1e-10
    assert abs(compute_distance_sq(a, b) - 25.0) < 1e-10
    print("[PASS] Distance computation")


def test_rate_uplink():
    """
    Test R_u formula against GA code.
    GA code: r_u_i = log2(1 + (theta * s[i]*P_s) / (d_su2**a2))
    where theta = e**(-E) * w_0 / xich_ma_u
    """
    # Use a sample UAV position
    uav_pos = np.array([10.0, 5.0, 8.0])
    d_su_sq = compute_distance_sq(uav_pos, cfg.W_S)

    # GA parameters
    theta_ga = math.exp(-cfg.EULER) * cfg.OMEGA_0 / cfg.SIGMA_U_SQ  # Same as THETA_0
    P_s_ga = cfg.P_S_MAX  # Using max power

    # GA formula (without bandwidth B, which is multiplied in the paper)
    r_u_ga = math.log2(1 + theta_ga * P_s_ga / (d_su_sq ** (cfg.ALPHA / 2)))

    # Our formula
    r_u_ours = compute_rate_uplink(d_su_sq, P_s_ga) / cfg.B_BANDWIDTH

    assert abs(r_u_ours - r_u_ga) < 1e-10, f"R_u mismatch: {r_u_ours} vs {r_u_ga}"
    print(f"[PASS] R_u formula (value: {r_u_ours:.4f} at d_su={math.sqrt(d_su_sq):.2f}m)")


def test_rate_downlink():
    """
    Test R_d formula against GA code.
    GA code: r_d_i1 = log2(1+(theta*(n_u*w_0*s[i]*P_s + P_u_bar*d_su2**a2)) / (d_su2**a2 * d_du2**a2))
    where P_u_bar = u[i]*P_u * (1 + ceil(xichma))
    """
    uav_pos = np.array([10.0, 5.0, 8.0])
    d_su_sq = compute_distance_sq(uav_pos, cfg.W_S)
    d_du_sq = compute_distance_sq(uav_pos, cfg.W_D)

    P_s_val = cfg.P_S_MAX
    P_u_val = cfg.P_U_MAX
    eta = cfg.ETA_MAX

    # GA formula
    theta_ga = math.exp(-cfg.EULER) * cfg.OMEGA_0 / cfg.SIGMA_U_SQ
    ceil_sigma = math.ceil(cfg.SIGMA_CACHE)  # = 1
    P_u_bar_ga = P_u_val * (1 + ceil_sigma)  # P_u * 2
    d_su_alpha = d_su_sq ** (cfg.ALPHA / 2)
    d_du_alpha = d_du_sq ** (cfg.ALPHA / 2)

    r_d_ga = math.log2(
        1 + theta_ga * (eta * cfg.OMEGA_0 * P_s_val + P_u_bar_ga * d_su_alpha)
        / (d_su_alpha * d_du_alpha)
    )

    # Our formula
    r_d_ours = compute_rate_downlink(d_su_sq, d_du_sq, P_s_val, P_u_val, eta) / cfg.B_BANDWIDTH

    assert abs(r_d_ours - r_d_ga) < 1e-10, f"R_d mismatch: {r_d_ours} vs {r_d_ga}"
    print(f"[PASS] R_d formula (value: {r_d_ours:.4f})")


def test_energy_fly():
    """
    Test E_fly formula against GA code.
    GA: e_fly_i = P_0 * (t + (k_1 / t) * dis**2) + P_1 * sqrt(sqrt(t**4 + k_2**2 * dis**4) -
                  k_2 * dis**2) + k_3 * dis**3 / t**2
    """
    q_curr = np.array([5.0, 5.0, 8.0])
    q_next = np.array([7.0, 6.0, 7.0])
    dis = compute_distance(q_curr, q_next)
    dt = cfg.DELTA_T

    # GA formula (manual)
    term1 = cfg.P_0 * (dt + cfg.K_1 / dt * dis ** 2)
    inner = math.sqrt(dt ** 4 + cfg.K_2 ** 2 * dis ** 4) - cfg.K_2 * dis ** 2
    term2 = cfg.P_1 * math.sqrt(max(inner, 0))
    term3 = cfg.K_3 * dis ** 3 / dt ** 2
    e_fly_ga = term1 + term2 + term3

    # Our formula
    e_fly_ours = compute_e_fly(q_curr, q_next)

    assert abs(e_fly_ours - e_fly_ga) < 1e-10, f"E_fly mismatch: {e_fly_ours} vs {e_fly_ga}"
    print(f"[PASS] E_fly formula (value: {e_fly_ours:.4f} mJ at dis={dis:.2f}m)")


def test_energy_harvested():
    """
    Test E_h formula against GA code.
    GA: e_h_i = micro*(1 - tau[i])*t*w_0*P_wpt/(distance(q0, w_s)**2)**a2
    """
    uav_pos = np.array([10.0, 5.0, 8.0])
    d_su_sq = compute_distance_sq(uav_pos, cfg.W_S)
    tau = 0.6
    dt = cfg.DELTA_T

    # GA formula
    d_su_alpha = d_su_sq ** (cfg.ALPHA / 2)
    e_h_ga = cfg.MU * (1 - tau) * dt * cfg.OMEGA_0 * cfg.P_WPT / d_su_alpha

    # Our formula
    e_h_ours = compute_energy_harvested(d_su_sq, tau)

    assert abs(e_h_ours - e_h_ga) < 1e-10, f"E_h mismatch: {e_h_ours} vs {e_h_ga}"
    print(f"[PASS] E_h formula (value: {e_h_ours:.4f} mJ)")


def test_environment_basic():
    """Test basic environment creation and stepping."""
    env = UAVEnvironment()

    state, _ = env.reset()
    assert state.shape == (env.state_dim,), f"State shape: {state.shape}"
    print(f"[PASS] Env reset, state dim: {env.state_dim}")

    # Take a random action
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    assert next_state.shape == (env.state_dim,)
    assert isinstance(float(reward), float)
    assert "throughput" in info
    print(f"[PASS] Env step, reward: {reward:.4f}, throughput: {info['throughput']:.4f}")


def test_full_episode():
    """Run a full episode with random actions and check results."""
    env = UAVEnvironment()
    state, _ = env.reset()
    total_reward = 0.0

    for _ in range(cfg.N_SLOTS):
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated:
            break

    results = env.get_results()
    print(f"[PASS] Full episode ({cfg.N_SLOTS} steps)")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Total throughput: {results['total_throughput']:.4f} Mbits")
    print(f"  Energy feasible: {results['energy_feasible']}")
    print(f"  Relay feasible: {results['relay_feasible']}")
    print(f"  Demand met: {results['demand_met']}")
    print(f"  Final positions: UAV1={results['final_positions'][0]}, UAV2={results['final_positions'][1]}")


def test_config_consistency():
    """Verify config parameters match GA code values."""
    # Check derived constants match
    assert abs(cfg.V_0 - math.sqrt(cfg.UAV_WEIGHT / (2 * cfg.RHO_AIR * cfg.ROTOR_AREA))) < 1e-10
    assert abs(cfg.DELTA_D - cfg.V_MAX * cfg.DELTA_T) < 1e-10
    assert cfg.N_SLOTS == int(cfg.T / cfg.DELTA_T)

    # Check GA code parameters
    assert abs(cfg.THETA_0 - math.exp(-cfg.EULER) * cfg.OMEGA_0 / cfg.SIGMA_U_SQ) < 1e-10
    print("[PASS] Config parameter consistency")


if __name__ == "__main__":
    print("=" * 60)
    print("UAV Environment Unit Tests")
    print("=" * 60)

    test_distance()
    test_config_consistency()
    test_rate_uplink()
    test_rate_downlink()
    test_energy_fly()
    test_energy_harvested()
    test_environment_basic()
    test_full_episode()

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
