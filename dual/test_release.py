"""
Diagnostic script to test sticky gripper release behavior.
Run from lerobot folder: python dual/test_release.py
"""

import numpy as np
import sys
import os

# Fix imports for your directory structure
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, script_dir)

def test_release_behavior():
    """Test that gripper properly releases cube."""
    print("=" * 60)
    print("TESTING GRIPPER RELEASE BEHAVIOR")
    print("=" * 60)
    
    # Import from your package structure
    from dual.envs import DualSO101PickCubeEnv
    
    env = DualSO101PickCubeEnv(render_mode=None)
    obs, info = env.reset(seed=42)
    
    print("\nGripper thresholds:")
    print("  CLOSED < {}".format(env.GRIPPER_CLOSED_THRESHOLD))
    print("  OPEN > {}".format(getattr(env, 'GRIPPER_OPEN_THRESHOLD', 'NOT SET - BUG!')))
    
    # Check if GRIPPER_OPEN_THRESHOLD exists
    if not hasattr(env, 'GRIPPER_OPEN_THRESHOLD'):
        print("\n[X] ERROR: GRIPPER_OPEN_THRESHOLD not found!")
        print("   You need to use the fixed environment file.")
        env.close()
        return False
    
    print("\n--- Simulating grasp-lift-release ---")
    
    print("\n[1] Initial state:")
    print("    Cube height: {:.3f}m".format(info['cube_height']))
    print("    Grasping arm: {}".format(env._get_grasping_arm()))
    
    # Force a grasp state for testing
    env.is_grasped_arm0 = True
    env.grasp_offset_arm0 = np.array([0, 0, 0])
    env.grasp_start_step = 0
    
    print("\n[2] After forcing grasp:")
    print("    Grasping arm: {}".format(env._get_grasping_arm()))
    print("    is_grasped_arm0: {}".format(env.is_grasped_arm0))
    
    # Now simulate opening the gripper
    print("\n[3] Testing release by opening gripper...")
    
    # Set gripper to open state (above GRIPPER_OPEN_THRESHOLD)
    env.data.qpos[5] = 0.8
    
    # Call the sticky gripper update
    env._apply_sticky_gripper()
    
    print("    Gripper value: {:.2f}".format(env.data.qpos[5]))
    print("    is_grasped_arm0 after update: {}".format(env.is_grasped_arm0))
    print("    Grasping arm: {}".format(env._get_grasping_arm()))
    
    success = not env.is_grasped_arm0
    
    if success:
        print("\n[OK] SUCCESS: Gripper properly released cube!")
    else:
        print("\n[X] FAIL: Gripper did not release cube!")
        print("   Check _apply_sticky_gripper() for GRIPPER_OPEN_THRESHOLD logic")
    
    env.close()
    return success


def test_reward_phases():
    """Test that reward function has proper placing incentives."""
    print("\n" + "=" * 60)
    print("TESTING REWARD STRUCTURE")
    print("=" * 60)
    
    from dual.envs import DualSO101PickCubeEnv
    
    env = DualSO101PickCubeEnv(render_mode=None)
    env.reset(seed=42)
    
    print("\nPhase 4 (PLACING) reward analysis:")
    
    test_cases = [
        {"cube_to_target": 0.2, "cube_height": 0.25, "released": False, 
         "desc": "Far from target, lifted, holding"},
        {"cube_to_target": 0.08, "cube_height": 0.25, "released": False, 
         "desc": "Near target, lifted, holding"},
        {"cube_to_target": 0.05, "cube_height": 0.21, "released": False, 
         "desc": "At target, lowering, holding"},
        {"cube_to_target": 0.05, "cube_height": 0.19, "released": True, 
         "desc": "At target, on table, RELEASED"},
    ]
    
    rewards = []
    
    for case in test_cases:
        # Create mock info dict
        TABLE_HEIGHT = 0.19
        info = {
            "cube_pos": np.array([env.target_pos[0], env.target_pos[1], case["cube_height"]]),
            "cube_height": case["cube_height"],
            "cube_to_target": case["cube_to_target"],
            "is_success": (case["cube_to_target"] < 0.06 and 
                          case["cube_height"] < TABLE_HEIGHT + 0.03 and
                          case["cube_height"] > TABLE_HEIGHT - 0.02),
            "is_at_target": case["cube_to_target"] < 0.06,
            "is_on_table": TABLE_HEIGHT - 0.02 < case["cube_height"] < TABLE_HEIGHT + 0.03,
            "dist0": 0.1,
            "dist1": 0.3,
            "min_dist_to_cube": 0.1,
            "arm0_ee_pos": np.array([0, 0, 0.3]),
            "arm1_ee_pos": np.array([0.2, 0, 0.3]),
            "any_contact": True,
            "contact0": True,
            "contact1": False,
            "primary_arm": 0,
            "grasping_arm": 0 if not case["released"] else None,
            "is_released": case["released"],
        }
        
        env.current_phase = 4
        
        if not case["released"]:
            env.is_grasped_arm0 = True
            env.data.qpos[5] = 0.2  # Closed
        else:
            env.is_grasped_arm0 = False
            env.data.qpos[5] = 0.8  # Open
        
        reward = env._compute_reward(info)
        rewards.append(reward)
        
        status = ">>>" if case["released"] and info["is_success"] else "   "
        print("  {} {}".format(status, case['desc']))
        print("       Reward: {:.2f}".format(reward))
    
    env.close()
    
    # Check if released-at-target has highest reward
    if rewards[-1] > max(rewards[:-1]):
        print("\n[OK] GOOD: Released at target has highest reward!")
    else:
        print("\n[!] WARNING: Released at target should have highest reward")
        print("   Got: {:.2f}, but max other: {:.2f}".format(rewards[-1], max(rewards[:-1])))


def test_full_episode():
    """Run a full episode with random actions to check for issues."""
    print("\n" + "=" * 60)
    print("TESTING FULL EPISODE")
    print("=" * 60)
    
    from dual.envs import DualSO101PickCubeEnv
    
    env = DualSO101PickCubeEnv(render_mode=None)
    obs, info = env.reset(seed=42)
    
    phase_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    max_height = 0.19
    total_reward = 0
    
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        
        phase_counts[info['current_phase']] += 1
        max_height = max(max_height, info['cube_height'])
        total_reward += reward
        
        if term:
            print("\n*** SUCCESS at step {}! ***".format(step))
            break
    
    print("\nEpisode summary:")
    print("  Total reward: {:.2f}".format(total_reward))
    print("  Max cube height: {:.3f}m".format(max_height))
    print("  Phase distribution:")
    phase_names = {0: "REACH", 1: "CONTACT", 2: "GRASP", 3: "LIFT", 4: "PLACE"}
    for phase, count in phase_counts.items():
        print("    Phase {} ({}): {} steps".format(phase, phase_names[phase], count))
    
    env.close()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DUAL-ARM ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)
    
    success1 = test_release_behavior()
    
    if success1:
        test_reward_phases()
        test_full_episode()
    else:
        print("\n[!] Fix the release behavior before continuing!")
        print("   Make sure dual_so101_env_fixed.py is in dual/envs/")