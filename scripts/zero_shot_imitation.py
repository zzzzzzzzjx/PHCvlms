#!/usr/bin/env python3
"""
Zero-shot imitation evaluation script for PHC model.
This script loads a trained model and performs zero-shot imitation on new motion data.
"""

import os
import sys
sys.path.append(os.getcwd())

import argparse
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Zero-shot imitation with PHC model")
parser.add_argument("--policy_path", type=str, required=True, 
                   help="Path to the trained policy checkpoint")
parser.add_argument("--motion_file", type=str, required=True,
                   help="Path to the motion .pkl file for imitation")
parser.add_argument("--action_offset_file", type=str, 
                   default="phc/data/action_offset_smpl.pkl",
                   help="Path to action offset file")
parser.add_argument("--humanoid_type", type=str, default="smpl", 
                   choices=["smpl", "smplx"], help="Type of humanoid model")
parser.add_argument("--num_envs", type=int, default=1, 
                   help="Number of environments to spawn")
parser.add_argument("--num_motions", type=int, default=10,
                   help="Number of motions to load from the motion library")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Launch the app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import joblib
from easydict import EasyDict

from phc.env.tasks.humanoid_im import HumanoidIm
from phc.learning.network_loader import load_mcp_mlp
from rl_games.algos_torch.players import rescale_actions
from phc.utils.config import set_np_formatting, set_seed

def main():
    """Main function for zero-shot imitation."""
    
    # Set up the environment
    env_cfg = {
        "env": {
            "numEnvs": args_cli.num_envs,
            "enableDebugVis": False,
            "num_levels": 1,
            "env_spacing": 5.0,
        },
        "task": {
            "name": "HumanoidIm",
            "randomize": False,
            "env": {
                "num_actions": 69,
                "num_observations": 1,
                "num_states": 1,
                "episode_length_s": 60.0,
                "mtm": False,
            },
            "motion_file": args_cli.motion_file,
            "models": [args_cli.policy_path],
            "num_prim": 3,
            "command": "run",
        },
        "sim": {
            "device": args_cli.device,
            "use_gpu_pipeline": True,
            "physx": {
                "num_threads": 4,
                "solver_type": 1,
                "use_gpu": True,
            }
        }
    }
    
    # Create environment
    env = HumanoidIm(config=EasyDict(env_cfg))
    
    # Load policy
    checkpoint = torch_ext.load_checkpoint(args_cli.policy_path)
    pnn = load_mcp_mlp(checkpoint, num_prim=3, has_lateral=False, 
                      activation="silu", device=args_cli.device)
    
    # Load action offset
    action_offset = joblib.load(args_cli.action_offset_file)
    pd_action_offset = action_offset[0]
    pd_action_scale = action_offset[1]
    
    # Get running mean and std
    running_mean = checkpoint['running_mean_std']['running_mean']
    running_var = checkpoint['running_mean_std']['running_var']
    
    # Reset environment
    obs_dict = env.reset()
    
    print("Starting zero-shot imitation...")
    print(f"Policy: {args_cli.policy_path}")
    print(f"Motion: {args_cli.motion_file}")
    print(f"Humanoid type: {args_cli.humanoid_type}")
    
    try:
        while True:
            # Get observations
            self_obs, task_obs = obs_dict["self_obs"], obs_dict["task_obs"]
            full_obs = torch.cat([self_obs, task_obs], dim=-1)
            
            # Normalize observations
            full_obs = ((full_obs - running_mean.float()) / 
                       torch.sqrt(running_var.float() + 1e-05))
            full_obs = torch.clamp(full_obs, min=-5.0, max=5.0)
            
            # Get actions from policy
            with torch.no_grad():
                actions, _ = pnn(full_obs, idx=0)
                actions = rescale_actions(-1, 1, torch.clamp(actions, -1, 1))
                actions = actions * pd_action_scale + pd_action_offset
                
            # Step the environment
            obs_dict, _, _, _, _ = env.step(actions)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    
    finally:
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    set_np_formatting()
    set_seed(42)
    main()