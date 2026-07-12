"""
System parameters for UAV trajectory optimization.
"""
import numpy as np
import math

# Time parameters
T = 30.0            # Total flight time (seconds)
DELTA_T = 0.5       # Time slot duration (seconds)
N_SLOTS = int(T / DELTA_T)  # Number of time slots (60)

# UAV speed
V_MAX = 20.0        # Maximum UAV speed (m/s)
DELTA_D = V_MAX * DELTA_T  # Max distance per slot (10 m)
A_MAX = 5.0         # Maximum acceleration (m/s^2)

# Spatial bounds
X_MAX = 20.0
Y_MAX = 10.0
Z_MIN = 3.0         # Minimum altitude
Z_MAX = 10.0        # Maximum altitude

# Node positions (3D column vectors as numpy arrays)
W_S = np.array([5.0, 0.0, 0.0])    # Source (BS) location
W_D = np.array([15.0, 0.0, 0.0])   # Destination location
MIDPOINT = np.array([10.0, 0.0, 5.0])  # V-shape midpoint (from BCD: midpoint of Source & Dest)

# UAV initial and final positions
Q_I1 = np.array([0.0, 10.0, 10.0])   # UAV 1 start
Q_F1 = np.array([20.0, 10.0, 10.0])  # UAV 1 end
Q_I2 = np.array([0.0, 10.0, 5.0])    # UAV 2 start
Q_F2 = np.array([20.0, 10.0, 5.0])   # UAV 2 end

NUM_UAVS = 2

# Channel parameters
ALPHA = 2.2           # Path loss exponent
OMEGA_0 = 1e-3        # Reference channel gain (-30 dB)
SIGMA_U_SQ = 1e-6     # AWGN noise variance at UBD
EULER = 0.5772156649  # Euler-Mascheroni constant
THETA_0 = math.exp(-EULER) * OMEGA_0 / SIGMA_U_SQ  # theta_0 for R_u
THETA = THETA_0       # theta for R_d (same expression)

# Power parameters (all in mW)
P_S_MAX = 10**1.6     # BS max transmit power (~39.81 mW, 16 dBm)
P_U_MAX = 5.0         # UAV max transmit power (5 mW)
P_B = 1e-3            # Backscatter circuit power consumption (mW)
P_WPT = 1e7           # WPT transmit power (10^4 W = 10^7 mW)

# Caching and backscatter
SIGMA_CACHE = 0.45    # Caching gain coefficient (c_r in paper)
ETA_MAX = 0.5         # Maximum backscatter coefficient
ETA = ETA_MAX         # We fix eta = eta_max (linear, no need to optimize)

# Bandwidth
# Note: GA and BCD code compute rates as log2(1+SNR) without multiplying
# by bandwidth B. We follow the same convention for comparable results.
B_BANDWIDTH = 1.0

# Energy harvesting
MU = 0.84             # EH efficiency

# Data demand
S_DEMAND = 20.0       # Demanded data (Mbits), variable in simulations

# Rotary-wing UAV energy model parameters (from paper Table II)
RHO_AIR = 1.225       # Air density (kg/m^3)
ROTOR_SOLIDITY = 0.05 # Rotor solidity
ROTOR_AREA = 0.8      # Rotor disc area (m^2)
OMEGA_BLADE = 100.0   # Blade angular velocity (rad/s)
ROTOR_RADIUS = 0.08   # Rotor radius (m)
UAV_WEIGHT = 0.5      # UAV weight (N)
DELTA_DRAG = 0.012    # Profile drag coefficient
I_INCR = 0.1          # Incremental correction factor
D_0 = 0.0151 / ROTOR_SOLIDITY / ROTOR_AREA  # Fuselage drag ratio

# Derived energy model constants
V_0 = math.sqrt(UAV_WEIGHT / (2 * RHO_AIR * ROTOR_AREA))  # Mean rotor induced velocity
P_0 = (DELTA_DRAG / 8) * RHO_AIR * ROTOR_SOLIDITY * ROTOR_AREA * (OMEGA_BLADE * ROTOR_RADIUS) ** 3
P_1 = (1 + I_INCR) * UAV_WEIGHT ** 1.5 / math.sqrt(2 * RHO_AIR * ROTOR_AREA)
K_1 = 3.0 / (OMEGA_BLADE ** 2 * ROTOR_RADIUS ** 2)
K_2 = 1.0 / (2 * V_0 ** 2)
K_3 = 0.5 * D_0 * RHO_AIR * ROTOR_SOLIDITY * ROTOR_AREA

# DDPG hyperparameters
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.99          # Discount factor
TAU_SOFT = 0.005      # Soft update coefficient for target networks
BUFFER_SIZE = 100000  # Replay buffer capacity
BATCH_SIZE = 256
NOISE_SIGMA = 0.5     # OU noise sigma (Increased for more exploration)
NOISE_THETA = 0.2     # OU noise theta
NOISE_DECAY = 0.999   # OU noise decay rate

# Training settings
MAX_EPISODES = 5000
MAX_STEPS = N_SLOTS   # Steps per episode = number of time slots
LOG_INTERVAL = 50
SAVE_INTERVAL = 500

# Reward shaping weights
LAMBDA_ENERGY = 50.0     # Penalty weight for energy constraint violation
LAMBDA_DATA_RELAY = 10.0 # Penalty weight for data relay constraint violation
LAMBDA_DEMAND = 20.0     # Penalty weight for demand not met at terminal
LAMBDA_BOUNDARY = 200.0  # Quadratic penalty for not reaching final position
THROUGHPUT_SCALE = 1.0   # Scale factor for throughput reward
LAMBDA_PROXIMITY = 8.0   # Bonus weight for flying close to Source and Destination
LAMBDA_MIDPOINT = 8.0    # Bonus weight for flying close to V-shape midpoint
LAMBDA_LOITER = 5.0      # Bonus for hovering close to midpoint
LOITER_RADIUS = 3.0      # Tight radius for loitering bonus (m)
LANDING_STEPS = 5        # Number of final steps where UAV is forced to fly toward destination

