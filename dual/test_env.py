"""
Test Dual-Arm SO-101 Gym Environment

Run from lerobot root:
    cd C:/Users/Gzw19/lerobot
    python dual/test_env.py
"""

import numpy as np
import time


def test_environment():
    print("\n" + "=" * 60)
    print("Testing Dual-Arm SO-101 Gym Environment")
    print("=" * 60 + "\n")
    
    # Import the environment
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from dual.envs.dual_so101_env import DualSO101PickCubeEnv
    
    # Create environment (no rendering for basic test)
    print("Creating environment...")
    env = DualSO101PickCubeEnv(render_mode=None)
    
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Number of actuators: {env.n_actuators}")
    
    # Test reset
    print("\nTesting reset...")
    obs, info = env.reset(seed=42)
    print(f"  Observation shape: {obs.shape}")
    print(f"  Initial cube pos: {info['cube_pos']}")
    print(f"  Initial target pos: {info['target_pos']}")
    
    # Test step
    print("\nTesting step with random actions...")
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
    print(f"  10 steps completed")
    print(f"  Total reward: {total_reward:.3f}")
    print(f"  Final cube height: {info['cube_height']:.3f}")
    
    # Test episode rollout
    print("\nTesting full episode rollout...")
    obs, info = env.reset()
    episode_reward = 0
    steps = 0
    
    start_time = time.time()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    elapsed = time.time() - start_time
    fps = steps / elapsed
    
    print(f"  Episode length: {steps} steps")
    print(f"  Episode reward: {episode_reward:.3f}")
    print(f"  Success: {info['is_success']}")
    print(f"  Simulation speed: {fps:.1f} FPS")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("ALL ENVIRONMENT TESTS PASSED!")
    print("=" * 60 + "\n")
    
    return True


def test_with_viewer():
    """Test environment with visual rendering."""
    print("\n" + "=" * 60)
    print("Testing with MuJoCo Viewer")
    print("=" * 60 + "\n")
    
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from dual.envs.dual_so101_env import DualSO101PickCubeEnv
    
    print("Creating environment with human rendering...")
    env = DualSO101PickCubeEnv(render_mode="human")
    
    obs, info = env.reset()
    print("Running 200 steps with random actions...")
    print("Close the viewer window to exit.\n")
    
    try:
        for i in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(0.02)  # Slow down for visualization
            
            if terminated or truncated:
                print(f"Episode ended at step {i+1}, resetting...")
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()
    
    print("Viewer test complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Test with visual rendering")
    args = parser.parse_args()
    
    # Basic tests first
    success = test_environment()
    
    if success and args.render:
        test_with_viewer()
    elif success:
        print("Run with --render to test visual rendering:")
        print("  python dual/test_env.py --render")