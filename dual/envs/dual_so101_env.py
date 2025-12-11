"""
Dual-Arm SO-101 Environment V12

Key fix: Robot hovers in place instead of transporting to target.
Problem: Holding cube in air gives guaranteed reward with zero risk.

Changes from V11:
1. HOVER PENALTY: Penalize staying still while holding cube above ground
2. DESCENT REWARD: When over target, reward lowering toward table
3. REDUCED LIFT REWARD: Once near target horizontally, stop rewarding height
4. STRONGER DIRECTIONAL SIGNAL: Increase transport gradient
5. VELOCITY BONUS: Reward moving (any direction initially, then toward target)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import os
from typing import Optional, Dict, Any


class DualSO101PickCubeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
        action_scale: float = 0.1,
        reward_type: str = "dense",
        xml_path: Optional[str] = None,
        use_multi_init: bool = True,
        use_curriculum: bool = True,
        curriculum_progress: float = 0.0,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.reward_type = reward_type
        self.use_multi_init = use_multi_init
        self.use_curriculum = use_curriculum
        self.curriculum_progress = curriculum_progress
        
        self.target_pos = np.array([0.1, 0.1, 0.19])
        self.current_step = 0
        
        self.was_lifted = False
        self.was_grasped = False
        self.max_lift_height = 0.0
        self.prev_cube_to_target = None
        self.prev_cube_pos = None  # NEW: track full position for velocity
        self.hover_steps = 0  # NEW: count steps hovering in place
        
        # Find XML
        if xml_path is None:
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "..", "assets", "dual_so101_scene.xml"),
                "dual/assets/dual_so101_scene.xml",
                os.path.join(os.path.dirname(__file__), "assets", "dual_so101_scene.xml"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    xml_path = path
                    break
            if xml_path is None:
                raise FileNotFoundError("Could not find dual_so101_scene.xml")
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Get cube DOF IDs
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_adr = self.model.jnt_dofadr[cube_joint_id]
        self.cube_dof_ids = list(range(cube_adr, cube_adr+6))
        self.cube_qpos_adr = self.model.jnt_qposadr[cube_joint_id]

        # Store joint limits
        self.joint_limits_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.joint_limits_high = self.model.actuator_ctrlrange[:, 1].copy()
        
        # Gripper range
        grip_min = self.joint_limits_low[5]
        grip_max = self.joint_limits_high[5]
        grip_range = grip_max - grip_min
        
        self.GRASP_DISTANCE = 0.055
        self.GRIPPER_CLOSE_THRESH = grip_min + 0.4 * grip_range
        self.GRIPPER_OPEN_THRESH = grip_min + 0.6 * grip_range
        self.RELEASE_DISTANCE = 0.10
        
        print(f"[ENV V12] Gripper range: [{grip_min:.2f}, {grip_max:.2f}]")
        print(f"[ENV V12] Close thresh: {self.GRIPPER_CLOSE_THRESH:.2f}, Open thresh: {self.GRIPPER_OPEN_THRESH:.2f}")
        
        # State
        self.is_grasped = False
        self.grasp_arm = None
        self.grasp_offset = None
        
        self.primary_arm = 0
        self.primary_arm_locked = False 
        self.current_phase = 0
        
        # IDs
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
        self.cube_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "cube_site")
        self.arm0_ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "arm0_ee_site")
        self.arm1_ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "arm1_ee_site")
        
        self.n_actuators = self.model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_actuators,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32)
        
        self.viewer = None
        self._render_context = None
        
        grip_max_val = grip_max
        self.base_init_pose = np.array([
            -0.38, 0.1, 0.05, -0.1, 0.0, grip_max_val,
            -0.38, 0.1, 0.05, -0.1, 0.0, grip_max_val
        ], dtype=np.float32)
        
        self.arm0_safe_ctrl = np.array([-0.5, 0.0, 0.0, 0.0, 0.0, grip_max_val])
        self.arm1_safe_ctrl = np.array([-0.5, 0.0, 0.0, 0.0, 0.0, grip_max_val])
        
        self.episode_count = 0
        self.success_count = 0

    def _get_ee_pos(self, arm_idx):
        site_id = self.arm0_ee_site_id if arm_idx == 0 else self.arm1_ee_site_id
        return self.data.site_xpos[site_id].copy()
    
    def _get_gripper_state(self, arm_idx):
        return self.data.qpos[5 if arm_idx == 0 else 11]
    
    def _is_gripper_closing(self, arm_idx):
        grip = self._get_gripper_state(arm_idx)
        return grip < self.GRIPPER_CLOSE_THRESH
    
    def _is_gripper_open(self, arm_idx):
        grip = self._get_gripper_state(arm_idx)
        return grip > self.GRIPPER_OPEN_THRESH

    def _update_grasp_state(self):
        cube_pos = self.data.site_xpos[self.cube_site_id].copy()
        
        if self.is_grasped:
            ee_pos = self._get_ee_pos(self.grasp_arm)
            dist = np.linalg.norm(ee_pos - cube_pos)
            
            if self._is_gripper_open(self.grasp_arm) or dist > self.RELEASE_DISTANCE:
                grip = self._get_gripper_state(self.grasp_arm)
                print(f"[DEBUG] Arm{self.grasp_arm} RELEASED! Grip={grip:.2f}, Dist={dist:.3f}")
                self.is_grasped = False
                self.grasp_arm = None
                self.grasp_offset = None
        else:
            for arm_idx in [self.primary_arm, 1 - self.primary_arm]:
                ee_pos = self._get_ee_pos(arm_idx)
                dist = np.linalg.norm(ee_pos - cube_pos)
                
                if self._is_gripper_closing(arm_idx) and dist < self.GRASP_DISTANCE:
                    grip = self._get_gripper_state(arm_idx)
                    print(f"[DEBUG] Arm{arm_idx} GRASP! Grip={grip:.2f}, Dist={dist:.3f}")
                    self.is_grasped = True
                    self.was_grasped = True
                    self.grasp_arm = arm_idx
                    self.grasp_offset = cube_pos - ee_pos
                    self.data.qvel[self.cube_dof_ids[0]:self.cube_dof_ids[0]+6] = 0
                    break

    def _apply_grasp_constraint(self):
        if not self.is_grasped or self.grasp_offset is None:
            self.data.xfrc_applied[self.cube_body_id, :] = 0
            return
        
        ee_pos = self._get_ee_pos(self.grasp_arm)
        self.grasp_offset *= 0.92
        target_pos = ee_pos + self.grasp_offset
        
        cube_qpos = self.data.qpos[self.cube_qpos_adr:self.cube_qpos_adr+3].copy()
        alpha = 0.3
        new_pos = cube_qpos + alpha * (target_pos - cube_qpos)
        new_pos[2] = max(new_pos[2], 0.19)
        
        self.data.qpos[self.cube_qpos_adr:self.cube_qpos_adr+3] = new_pos
        self.data.qvel[self.cube_dof_ids[0]:self.cube_dof_ids[0]+6] *= 0.5
        self.data.xfrc_applied[self.cube_body_id, 2] = 0.05 * 9.81

    def _select_primary_arm(self, info):
        if self.primary_arm_locked:
            return
        
        if self.is_grasped:
            self.primary_arm = self.grasp_arm
            self.primary_arm_locked = True
            return
        
        if info["dist0"] < info["dist1"]:
            self.primary_arm = 0
        else:
            self.primary_arm = 1
        
        if info["min_dist"] < 0.08:
            self.primary_arm_locked = True

    def _mask_action(self, action):
        masked_action = action.copy()
        if self.primary_arm == 0:
            masked_action[6:12] = 0.0
        else:
            masked_action[0:6] = 0.0
        return masked_action

    def _retract_non_primary_arm(self):
        retract_speed = 0.02
        if self.primary_arm == 0:
            current = self.data.ctrl[6:12]
            target = self.arm1_safe_ctrl
            diff = target - current
            self.data.ctrl[6:12] = current + np.clip(diff, -retract_speed, retract_speed)
        else:
            current = self.data.ctrl[0:6]
            target = self.arm0_safe_ctrl
            diff = target - current
            self.data.ctrl[0:6] = current + np.clip(diff, -retract_speed, retract_speed)

    def _update_phase(self, info):
        cube_height = info["cube_height"]
        cube_to_target = info["cube_to_target"]
        TABLE_HEIGHT = 0.19
        
        if self.current_phase == 0:
            if info["min_dist"] < 0.06:
                self.current_phase = 1
        elif self.current_phase == 1:
            if self.is_grasped:
                self.current_phase = 2
        elif self.current_phase == 2:
            if cube_height > TABLE_HEIGHT + 0.02:
                self.current_phase = 3
        elif self.current_phase == 3:
            if cube_to_target < 0.08:
                self.current_phase = 4
        elif self.current_phase == 4:
            if cube_height < TABLE_HEIGHT + 0.03:
                self.current_phase = 5
        
        if self.current_phase >= 2 and not self.is_grasped and cube_height < TABLE_HEIGHT + 0.01:
            if cube_to_target > 0.1:
                self.current_phase = 0
                self.primary_arm_locked = False

    def _get_obs(self):
        qpos = self.data.qpos[:12].copy()
        qvel = self.data.qvel[:12].copy()
        cube_pos = self.data.site_xpos[self.cube_site_id].copy()
        arm0_ee = self.data.site_xpos[self.arm0_ee_site_id].copy()
        arm1_ee = self.data.site_xpos[self.arm1_ee_site_id].copy()
        
        grip0_raw = qpos[5]
        grip1_raw = qpos[11]
        grip_min = self.joint_limits_low[5]
        grip_max = self.joint_limits_high[5]
        grip0_norm = (grip0_raw - grip_min) / (grip_max - grip_min + 1e-6)
        grip1_norm = (grip1_raw - grip_min) / (grip_max - grip_min + 1e-6)
        
        target_dist = np.linalg.norm(cube_pos[:2] - self.target_pos[:2])
        is_p0 = 1.0 if self.primary_arm == 0 else 0.0
        is_p1 = 1.0 if self.primary_arm == 1 else 0.0
        
        obs = np.concatenate([
            qpos, qvel, cube_pos, arm0_ee, arm1_ee,
            [grip0_norm, grip1_norm],
            [self.current_phase / 5.0],  # Now 5 phases
            [target_dist],
            [1.0 if self.is_grasped and self.grasp_arm == 0 else 0.0,
             1.0 if self.is_grasped and self.grasp_arm == 1 else 0.0],
            self.target_pos,
            [is_p0, is_p1],
        ]).astype(np.float32)
        return obs

    def _get_info(self):
        cube_pos = self.data.site_xpos[self.cube_site_id].copy()
        ee0 = self.data.site_xpos[self.arm0_ee_site_id].copy()
        ee1 = self.data.site_xpos[self.arm1_ee_site_id].copy()
        dist0 = np.linalg.norm(ee0 - cube_pos)
        dist1 = np.linalg.norm(ee1 - cube_pos)
        
        cube_to_target = np.linalg.norm(cube_pos[:2] - self.target_pos[:2])
        
        TABLE_HEIGHT = 0.19
        lift_amount = cube_pos[2] - TABLE_HEIGHT
        if lift_amount > self.max_lift_height:
            self.max_lift_height = lift_amount
        if lift_amount > 0.02: 
            self.was_lifted = True
        
        is_at_target = cube_to_target < 0.06
        is_on_table = 0.17 < cube_pos[2] < 0.25
        is_success = is_at_target and is_on_table and self.was_lifted and self.was_grasped
        
        # Calculate velocity (for hover detection)
        cube_velocity = 0.0
        if self.prev_cube_pos is not None:
            cube_velocity = np.linalg.norm(cube_pos - self.prev_cube_pos)
        
        return {
            "dist0": dist0, "dist1": dist1,
            "min_dist": min(dist0, dist1),
            "cube_pos": cube_pos,
            "cube_height": cube_pos[2],
            "cube_to_target": cube_to_target,
            "cube_velocity": cube_velocity,
            "is_success": is_success,
            "is_on_table": is_on_table,
            "is_at_target": is_at_target,
            "was_lifted": self.was_lifted,
            "was_grasped": self.was_grasped,
            "max_lift": self.max_lift_height,
            "ee0": ee0, "ee1": ee1,
            "primary_arm": self.primary_arm,
        }

    def _compute_reward(self, info, action):
        if self.reward_type == "sparse":
            return 100.0 if info["is_success"] else 0.0
        
        reward = 0.0
        TABLE_HEIGHT = 0.19
        cube_height = info["cube_height"]
        cube_to_target = info["cube_to_target"]
        cube_velocity = info["cube_velocity"]
        lift_amount = cube_height - TABLE_HEIGHT
        min_dist = info["min_dist"]
        
        # === PHASE 0: REACHING (max ~4) ===
        reach_reward = 3.0 * np.exp(-5.0 * min_dist)
        reward += reach_reward
        
        if min_dist < 0.10:
            reward += 0.5
        if min_dist < 0.06:
            reward += 0.5
        
        # === PHASE 1: GRIPPER CLOSING (max ~2) ===
        if min_dist < 0.08:
            grip = self._get_gripper_state(self.primary_arm)
            grip_min = self.joint_limits_low[5]
            grip_max = self.joint_limits_high[5]
            grip_normalized = 1.0 - (grip - grip_min) / (grip_max - grip_min + 1e-6)
            close_reward = 2.0 * max(0, grip_normalized)
            reward += close_reward
        
        # === PHASE 2: GRASPING (flat bonus) ===
        if self.is_grasped:
            reward += 2.0
            
        # === PHASE 3: LIFTING (CAPPED) ===
        if self.is_grasped and lift_amount > 0.01:
            lift_capped = min(lift_amount, 0.05) 
            lift_reward = 3.0 * (lift_capped / 0.05) 
            reward += lift_reward
            
        # === PHASE 4: TRANSPORT ===
        if self.is_grasped and lift_amount > 0.02:
            
            # 4a. DIRECTIONAL REWARD
            if self.prev_cube_to_target is not None:
                delta = self.prev_cube_to_target - cube_to_target
                reward += 300.0 * delta
            
            # 4b. ESCALATING HOVER PENALTY
            if cube_velocity < 0.01: 
                self.hover_steps += 1
            else:
                self.hover_steps = max(0, self.hover_steps - 2)
            
            # FIX: Escalating penalty - gets worse the longer you hover
            if self.hover_steps > 30:
                reward -= 0.5 * (self.hover_steps - 30)  # -0.5, -1.0, -1.5, ...
                
            # 4c. DISTANCE BASE (reduced to make descent more attractive)
            transport_base = 3.0 * np.exp(-4.0 * cube_to_target)  # Reduced from 5.0
            reward += transport_base

            # 4d. Milestone bonuses
            if cube_to_target < 0.15:
                reward += 2.0
            if cube_to_target < 0.10:
                reward += 3.0
            if cube_to_target < 0.07:
                reward += 5.0
        
        # === PHASE 5: DESCENT & PLACING (STRENGTHENED) ===
        if self.is_grasped and cube_to_target < 0.12:
            # FIX: Strong descent gradient - reward being lower when close to target
            ideal_height = TABLE_HEIGHT + 0.02  # 0.21m
            
            if cube_height > ideal_height:
                # Reward for descending: more reward the closer to table
                descent_reward = 10.0 * max(0, 0.35 - cube_height)  # Max ~1.4 at height 0.21
                reward += descent_reward
                
                # Extra bonus for getting really low while over target
                if cube_height < 0.25:
                    reward += 5.0
                if cube_height < 0.22:
                    reward += 10.0
            else:
                # At correct height - big bonus
                reward += 15.0
        
        # === PHASE 6: RELEASE & SUCCESS ===
        if cube_to_target < 0.06:
            # FIX: Reward for opening gripper when in position
            if self.is_grasped and cube_height < 0.25:
                # Encourage gripper opening when close and low
                grip = self._get_gripper_state(self.grasp_arm)
                grip_min = self.joint_limits_low[5]
                grip_max = self.joint_limits_high[5]
                grip_openness = (grip - grip_min) / (grip_max - grip_min + 1e-6)
                
                # Reward opening the gripper (grip_openness closer to 1 = more open)
                if grip_openness > 0.5:
                    reward += 10.0 * grip_openness  # Up to 10 for fully open
            
            if info["is_on_table"] and self.was_lifted:
                if not self.is_grasped:  # Successfully placed
                    reward += 50.0  # Increased from 30.0
        
        # === PENALTIES ===
        # Pushing cube without grasping
        if not self.was_lifted and cube_to_target < 0.1:
            reward -= 2.0
        
        # FIX: Penalty for being too high when close to target
        if self.is_grasped and cube_to_target < 0.10 and cube_height > 0.30:
            reward -= 3.0  # Punish flying high when you should be descending
        
        # Time penalty
        reward -= 0.03
        
        # === SUCCESS ===
        if info["is_success"]:
            time_bonus = max(0, 50 * (1 - self.current_step / self.max_episode_steps))
            reward += 100.0 + time_bonus
        
        # Update tracking
        self.prev_cube_to_target = cube_to_target
        self.prev_cube_pos = info["cube_pos"].copy()
            
        return reward

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_count += 1
        self.current_phase = 0
        self.primary_arm_locked = False
        
        self.was_lifted = False
        self.was_grasped = False
        self.max_lift_height = 0.0
        self.prev_cube_to_target = None
        self.prev_cube_pos = None
        self.hover_steps = 0
        
        self.is_grasped = False
        self.grasp_arm = None
        self.grasp_offset = None
        
        mujoco.mj_resetData(self.model, self.data)
        self.data.xfrc_applied[self.cube_body_id, :] = 0
        
        grip_max = self.joint_limits_high[5]
        
        if self.use_multi_init and self.np_random is not None:
            noise = self.np_random.uniform(-0.05, 0.05, size=12)
            init_pose = self.base_init_pose.copy() + noise
        else:
            init_pose = self.base_init_pose.copy()
        
        init_pose[5] = grip_max   
        init_pose[11] = grip_max  
        
        self.data.qpos[:12] = init_pose
        self.data.ctrl[:] = init_pose
        
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_adr = self.model.jnt_qposadr[cube_joint_id]
        
        if self.use_curriculum and self.np_random is not None:
            difficulty = self.np_random.random()
            easy_prob = 0.3 * (1 - self.curriculum_progress)
            if difficulty < easy_prob:
                self.data.qpos[cube_adr] = self.target_pos[0] + self.np_random.uniform(-0.02, 0.02)
                self.data.qpos[cube_adr+1] = self.target_pos[1] + self.np_random.uniform(-0.02, 0.02)
            else:
                self.data.qpos[cube_adr] = self.np_random.uniform(-0.03, 0.03)
                self.data.qpos[cube_adr+1] = self.np_random.uniform(-0.03, 0.03)
        else:
            self.data.qpos[cube_adr] = np.random.uniform(-0.03, 0.03)
            self.data.qpos[cube_adr+1] = np.random.uniform(-0.03, 0.03)
        
        self.data.qpos[cube_adr+2] = 0.19
        
        mujoco.mj_forward(self.model, self.data)
        
        info = self._get_info()
        self._select_primary_arm(info)
        
        self.prev_cube_to_target = info["cube_to_target"]
        self.prev_cube_pos = info["cube_pos"].copy()
        
        return self._get_obs(), info

    def step(self, action):
        self.current_step += 1
        action = np.clip(action, -1.0, 1.0)
        
        info = self._get_info()
        self._select_primary_arm(info)
        
        masked_action = self._mask_action(action)
        
        current_ctrl = self.data.ctrl.copy()
        target_ctrl = current_ctrl + (masked_action * self.action_scale)
        target_ctrl = np.clip(target_ctrl, self.joint_limits_low, self.joint_limits_high)
        self.data.ctrl[:] = target_ctrl
        
        self._retract_non_primary_arm()
        
        for _ in range(10):
            self._update_grasp_state()
            self._apply_grasp_constraint()
            mujoco.mj_step(self.model, self.data)
            
        # Debug prints
        if self.current_step % 25 == 0:
            cube_pos = self.data.site_xpos[self.cube_site_id]
            
            ee_pos = self._get_ee_pos(self.primary_arm)
            dist = np.linalg.norm(ee_pos - cube_pos)
            grip = self._get_gripper_state(self.primary_arm)
            
            held_status = f"ARM{self.grasp_arm}" if self.is_grasped else "NONE"
            cube_to_tgt = np.linalg.norm(cube_pos[:2] - self.target_pos[:2])
            
            # Direction indicator
            direction = ""
            if self.prev_cube_to_target is not None:
                delta = self.prev_cube_to_target - cube_to_tgt
                if delta > 0.005:
                    direction = "→TGT"
                elif delta < -0.005:
                    direction = "←AWAY"
                else:
                    direction = f"~(hov:{self.hover_steps})"
            
            print(f"[{self.current_step}] HELD: {held_status} | Dist: {dist:.3f} | Grip: {grip:.2f} | CubeZ: {cube_pos[2]:.3f} | ToTarget: {cube_to_tgt:.3f} {direction}")

        info = self._get_info()
        self._update_phase(info)
        
        reward = self._compute_reward(info, masked_action)
        obs = self._get_obs()
        
        terminated = info["is_success"]
        if terminated:
            self.success_count += 1
        truncated = self.current_step >= self.max_episode_steps
        
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            if self._render_context is None:
                self._render_context = mujoco.Renderer(self.model, height=480, width=640)
            self._render_context.update_scene(self.data)
            return self._render_context.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        if self._render_context:
            self._render_context.close()
            self._render_context = None


if __name__ == "__main__":
    print("Testing V12 - Anti-hover reward shaping...")
    env = DualSO101PickCubeEnv(render_mode=None)
    
    print(f"\n=== Observation space: {env.observation_space.shape} ===")
    
    print("\n=== Test: Hover penalty accumulation ===")
    obs, info = env.reset(seed=42)
    print(f"Initial cube_to_target: {info['cube_to_target']:.3f}")
    
    # Simulate hovering
    for i in range(100):
        action = np.zeros(12)  # No movement
        obs, reward, term, trunc, info = env.step(action)
    print(f"After 100 no-op steps, hover_steps: {env.hover_steps}")
    
    env.close()
    print("\nDone!")