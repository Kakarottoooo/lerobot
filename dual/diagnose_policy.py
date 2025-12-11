"""
Diagnosis script that checks ACTUAL applied controls, not raw model outputs.

The V7 environment masks actions internally, so we need to check:
1. What the model outputs (raw actions)
2. What actually gets applied (after masking)

python dual/diagnose_policy.py --model dual/checkpoints/new/final_model.zip
"""

import os
import sys
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from dual.envs.dual_so101_env import DualSO101PickCubeEnv


def diagnose(model_path, n_episodes=5):
    print("="*60)
    print("POLICY DIAGNOSIS V2")
    print("="*60)
    
    # Load environment WITHOUT curriculum (standard start)
    env = DualSO101PickCubeEnv(render_mode="human", use_curriculum=False)
    
    # Check if environment has action masking
    has_masking = hasattr(env, '_mask_action')
    print(f"\nEnvironment has action masking: {has_masking}")
    if not has_masking:
        print("⚠️  WARNING: Using old environment without action masking!")
    
    # Load model
    model = SAC.load(model_path)
    
    total_successes = 0
    
    for ep in range(n_episodes):
        print(f"\n--- Episode {ep} ---")
        obs, info = env.reset()
        
        # Track metrics
        phase_times = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        
        # Track RAW model outputs
        raw_arm0_active = 0
        raw_arm1_active = 0
        raw_both_active = 0
        
        # Track ACTUAL applied controls
        applied_arm0_active = 0
        applied_arm1_active = 0
        applied_both_active = 0
        
        cube_dropped = False
        cube_off_table = False
        max_height = 0.19
        step = 0
        
        prev_ctrl = env.data.ctrl.copy()
        
        for step in range(500):
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Analyze RAW model output
            raw_arm0_mag = np.linalg.norm(action[0:6])
            raw_arm1_mag = np.linalg.norm(action[6:12])
            
            if raw_arm0_mag > 0.3:
                raw_arm0_active += 1
            if raw_arm1_mag > 0.3:
                raw_arm1_active += 1
            if raw_arm0_mag > 0.3 and raw_arm1_mag > 0.3:
                raw_both_active += 1
            
            # Step environment
            obs, reward, term, trunc, info = env.step(action)
            
            # Check ACTUAL control changes (what was really applied)
            curr_ctrl = env.data.ctrl.copy()
            ctrl_change = curr_ctrl - prev_ctrl
            
            applied_arm0_mag = np.linalg.norm(ctrl_change[0:6])
            applied_arm1_mag = np.linalg.norm(ctrl_change[6:12])
            
            if applied_arm0_mag > 0.01:
                applied_arm0_active += 1
            if applied_arm1_mag > 0.01:
                applied_arm1_active += 1
            if applied_arm0_mag > 0.01 and applied_arm1_mag > 0.01:
                applied_both_active += 1
            
            prev_ctrl = curr_ctrl.copy()
            
            env.render()
            
            # Track phase
            phase_times[env.current_phase] += 1
            
            # Track cube
            if info["cube_height"] > max_height:
                max_height = info["cube_height"]
            if info["cube_height"] < 0.1:
                cube_dropped = True
            if np.linalg.norm(info["cube_pos"][:2]) > 0.2:
                cube_off_table = True
            
            if term or trunc:
                break
        
        total_steps = max(step, 1)
        
        # Report
        is_success = info.get('is_success', False)
        if is_success:
            total_successes += 1
            
        print(f"  Success: {is_success}")
        print(f"  Steps: {step + 1}")
        print(f"  Final phase: {env.current_phase}")
        print(f"  Phase distribution: {phase_times}")
        print(f"  Max cube height: {max_height:.3f}m")
        print(f"  Cube dropped: {cube_dropped}")
        print(f"  Cube off table: {cube_off_table}")
        print(f"  Primary arm: {env.primary_arm} (locked: {env.primary_arm_locked})")
        
        print(f"\n  RAW Model Output (before masking):")
        print(f"    Arm0 active: {raw_arm0_active} steps ({100*raw_arm0_active/total_steps:.1f}%)")
        print(f"    Arm1 active: {raw_arm1_active} steps ({100*raw_arm1_active/total_steps:.1f}%)")
        print(f"    BOTH active: {raw_both_active} steps ({100*raw_both_active/total_steps:.1f}%)")
        
        print(f"\n  ACTUAL Applied Controls (after masking):")
        print(f"    Arm0 moving: {applied_arm0_active} steps ({100*applied_arm0_active/total_steps:.1f}%)")
        print(f"    Arm1 moving: {applied_arm1_active} steps ({100*applied_arm1_active/total_steps:.1f}%)")
        print(f"    BOTH moving: {applied_both_active} steps ({100*applied_both_active/total_steps:.1f}%)")
        
        # Warnings based on ACTUAL applied controls
        if applied_both_active > total_steps * 0.3:
            print("  ⚠️  WARNING: Both arms actually moving >30% - masking not working!")
        if cube_dropped:
            print("  ⚠️  WARNING: Cube dropped off table!")
        if cube_off_table:
            print("  ⚠️  WARNING: Cube pushed off workspace!")
        if phase_times[0] > total_steps * 0.8:
            print("  ⚠️  WARNING: Stuck in reaching phase")
    
    print("\n" + "="*60)
    print(f"OVERALL: {total_successes}/{n_episodes} successes ({100*total_successes/n_episodes:.1f}%)")
    print("="*60)
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", "-n", type=int, default=5)
    args = parser.parse_args()
    diagnose(args.model, args.episodes)