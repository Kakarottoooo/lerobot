"""
Visual diagnostic to see what the arm is doing.
Shows arm positions and distances in real-time.

python dual/diagnose_reaching.py --model dual/checkpoints/ss/best_model.zip
"""

import os
import sys
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from dual.envs.dual_so101_env import DualSO101PickCubeEnv


def diagnose_reaching(model_path, n_episodes=2):
    print("="*60)
    print("REACHING DIAGNOSTIC")
    print("="*60)
    
    env = DualSO101PickCubeEnv(render_mode="human", use_curriculum=False)
    model = SAC.load(model_path)
    
    for ep in range(n_episodes):
        print(f"\n{'='*60}")
        print(f"EPISODE {ep}")
        print(f"{'='*60}")
        
        obs, info = env.reset()
        
        cube_pos = info['cube_pos']
        print(f"Cube position: {cube_pos}")
        print(f"Target position: {env.target_pos}")
        
        initial_dist0 = info['dist0']
        initial_dist1 = info['dist1']
        print(f"Initial distances: arm0={initial_dist0:.3f}m, arm1={initial_dist1:.3f}m")
        
        min_dist0 = initial_dist0
        min_dist1 = initial_dist1
        
        for step in range(300):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            env.render()
            
            # Track minimum distances achieved
            if info['dist0'] < min_dist0:
                min_dist0 = info['dist0']
            if info['dist1'] < min_dist1:
                min_dist1 = info['dist1']
            
            # Print progress every 50 steps
            if step % 50 == 0:
                print(f"\nStep {step}:")
                print(f"  Primary arm: {env.primary_arm} (locked: {env.primary_arm_locked})")
                print(f"  Arm0 dist: {info['dist0']:.3f}m (min: {min_dist0:.3f}m)")
                print(f"  Arm1 dist: {info['dist1']:.3f}m (min: {min_dist1:.3f}m)")
                print(f"  EE0 pos: {info['ee0']}")
                print(f"  EE1 pos: {info['ee1']}")
                print(f"  Cube pos: {info['cube_pos']}")
                
                # Check if arm is moving toward cube
                if env.primary_arm == 0:
                    progress = initial_dist0 - info['dist0']
                else:
                    progress = initial_dist1 - info['dist1']
                print(f"  Progress toward cube: {progress:.3f}m")
            
            if term or trunc:
                break
        
        print(f"\n--- Final Stats ---")
        print(f"Best distance achieved:")
        print(f"  Arm0: {min_dist0:.3f}m (improved by {initial_dist0 - min_dist0:.3f}m)")
        print(f"  Arm1: {min_dist1:.3f}m (improved by {initial_dist1 - min_dist1:.3f}m)")
        
        if min_dist0 > 0.15 and min_dist1 > 0.15:
            print("  ⚠️ ARM NEVER GOT CLOSE TO CUBE!")
        elif min(min_dist0, min_dist1) > 0.08:
            print("  ⚠️ Got closer but never reached lock distance (0.08m)")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    diagnose_reaching(args.model)