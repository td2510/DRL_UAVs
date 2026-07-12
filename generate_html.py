import numpy as np
import plotly.graph_objects as go
import config as cfg
from uav_env import UAVEnvironment
from ddpg_agent import DDPGAgent
import os

def generate_interactive_plot():
    # Load environment and model
    env = UAVEnvironment()
    agent = DDPGAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    model_path = os.path.join(os.path.dirname(__file__), "results", "best_feasible_model.pth")
    agent.load(model_path)
    
    # Simulate one episode
    state, _ = env.reset()
    for step in range(cfg.N_SLOTS):
        action = agent.predict(state, add_noise=False)
        state, _, done, _, _ = env.step(action)
        if done: break
        
    results = env.get_results()
    trajectories = results['trajectories']
    
    # Create interactive plot
    fig = go.Figure()
    
    colors = ['blue', 'orange']
    for i, traj in enumerate(trajectories):
        # The line and markers
        fig.add_trace(go.Scatter3d(
            x=traj[:,0], y=traj[:,1], z=traj[:,2],
            mode='lines+markers',
            name=f'UAV {i+1}',
            marker=dict(size=4, color=colors[i]),
            line=dict(color=colors[i], width=4)
        ))
        
        # Start point
        fig.add_trace(go.Scatter3d(
            x=[traj[0,0]], y=[traj[0,1]], z=[traj[0,2]],
            mode='markers', name=f'UAV {i+1} Start',
            marker=dict(size=8, color='green')
        ))
        
        # End point
        fig.add_trace(go.Scatter3d(
            x=[traj[-1,0]], y=[traj[-1,1]], z=[traj[-1,2]],
            mode='markers', name=f'UAV {i+1} End',
            marker=dict(size=8, color='red')
        ))

    # Add Source and Destination
    fig.add_trace(go.Scatter3d(
        x=[cfg.W_S[0]], y=[cfg.W_S[1]], z=[cfg.W_S[2]],
        mode='markers+text', name='Source',
        text=['Source'], textposition='top center',
        marker=dict(size=10, symbol='diamond', color='red')
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[cfg.W_D[0]], y=[cfg.W_D[1]], z=[cfg.W_D[2]],
        mode='markers+text', name='Destination',
        text=['Destination'], textposition='top center',
        marker=dict(size=10, symbol='square', color='blue')
    ))

    # Layout
    fig.update_layout(
        title='Interactive 3D UAV Trajectory (DRL)',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            xaxis=dict(range=[0, cfg.X_MAX]),
            yaxis=dict(range=[0, cfg.Y_MAX]),
            zaxis=dict(range=[0, cfg.Z_MAX])
        ),
        width=1000,
        height=800
    )
    
    save_path = os.path.join(os.path.dirname(__file__), "results", "trajectory_3d.html")
    fig.write_html(save_path)
    print(f"Interactive 3D plot saved to {save_path}")

if __name__ == "__main__":
    generate_interactive_plot()
