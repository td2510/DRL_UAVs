# DRL-based UAV Trajectory Optimization

This project implements a Deep Reinforcement Learning (DRL) approach using Deep Deterministic Policy Gradient (DDPG) to optimize the trajectories of multiple UAVs. The goal is to maximize data throughput while satisfying energy constraints, data relay constraints, and specific starting/landing position requirements.

## Requirements

Install the dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `config.py`: Configuration file for environment parameters, physics limits, and reward shaping weights.
- `uav_env.py`: Custom Gym-like environment defining the UAV physics (acceleration-based), throughput calculations, and reward logic.
- `ddpg_agent.py`: Implementation of the DDPG algorithm (Actor-Critic networks, Replay Buffer, OUNoise).
- `train.py`: Main script to train the UAV DRL agent.
- `test.py`: Script to evaluate a trained model and generate a 2D trajectory plot.
- `generate_html.py`: Script to generate an interactive 3D Plotly visualization of the trajectories.

## How to Run

1. Train the Model
Run the training script to train the DDPG agent. By default, it runs for 5000 episodes:
```bash
python train.py --episodes 5000
```
During training, the best feasible model will be automatically saved to `results/best_feasible_model.pth`.

2. Evaluate the Model
After training, you can evaluate the agent and generate a 2D trajectory image:
```bash
python test.py --num-episodes 1
```
The plot will be saved in the `results/` folder.

3. Generate Interactive 3D Plot
To visualize the final trajectories in an interactive 3D environment:
```bash
python generate_html.py
```
Open `results/trajectory_3d.html` in any web browser to view and rotate the 3D map.

## Key Features

- Acceleration-based Control: Action outputs define UAV accelerations instead of direct velocities, leading to smoother trajectories.
- Reward Shaping: Includes proximity bonuses, a time-weighted midpoint bonus to create V-shaped paths, and a loitering bonus to encourage UAVs to hover near optimal throughput locations.
- Feasibility Tracking: Only saves the model if it successfully reaches the destination and meets all energy/data constraints.
