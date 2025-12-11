import sys
import os
import time
import mujoco.viewer
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dual.envs.dual_so101_env import DualSO101PickCubeEnv

def debug_pose():
    # Create environment with "human" render mode
    env = DualSO101PickCubeEnv(render_mode="human")
    obs, info = env.reset()

    print("Checking Initial Pose...")
    print(f"Target Shoulder Lift in Code: {env.init_qpos[1]}")
    print("Press Ctrl+C in terminal to exit.")

    # Just render the robot. DO NOT STEP.
    # This allows you to see exactly how it spawns before physics kicks in.
    while True:
        env.render()
        time.sleep(0.01)

if __name__ == "__main__":
    debug_pose()