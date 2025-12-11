"""
Dual-Arm SO-101 Environment V3 - Fixed Placing & Release

Key Fixes from V2:
1. Sticky gripper RELEASES when gripper opens (was never releasing)
2. Strong Phase 4 PLACING rewards
3. Time penalty for holding cube too long (forces release)
4. Bonus for releasing cube at target location
5. Single-arm enforcement (other arm stays away)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import os
import time
from typing import Optional, Dict, Any, Tuple


class DualSO101PickCubeEnv(gym.Env):
    """
    Dual-arm SO-101 environment with proper pick-AND-PLACE.
    
    The key insight: V2 rewarded lifting but not releasing.
    This version adds strong incentives to PLACE the cube at target.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
        action_scale: float = 0.1,
        reward_type: str = "dense",
        target_height: float = 0.22,
        xml_path: Optional[str] = None,
        use_multi_init: bool = True,
        use_curriculum: bool = False,
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.reward_type = reward_type
        self.target_height = target_height
        self.current_step = 0
        self.use_multi_init = use_multi_init
        self.use_curriculum = use_curriculum
        
        # Single-arm grasping control
        self.grasping_arm = None
        self.target_pos = np.array([0.1, 0.1, 0.19])  # Target placement location

        # Safe position for non-grasping arm
        self.arm0_safe_pos = np.array([-0.2, 0.15, 0.25])
        self.arm1_safe_pos = np.array([0.2, 0.15, 0.25])
        
        # Find XML Path
        if xml_path is None:
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "..", "assets", "dual_so101_scene.xml"),
                "dual/assets/dual_so101_scene.xml",
                "assets/dual_so101_scene.xml",
                "dual_so101_scene.xml",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    xml_path = path
                    break
            if xml_path is None:
                raise FileNotFoundError("Could not find dual_so101_scene.xml")
        
        # Load Model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # === STICKY GRIPPER CONFIG ===
        self.GRASP_THRESHOLD = 0.05
        self.GRIPPER_CLOSED_THRESHOLD = 0.4  # Below this = closed
        self.GRIPPER_OPEN_THRESHOLD = 0.6    # Above this = open (releases cube)
        self.STICKY_STRENGTH = 8.0           # Reduced from 10
        self.MAX_FORCE = 2.0

        # Sticky gripper state
        self.is_grasped_arm0 = False
        self.is_grasped_arm1 = False
        self.grasp_offset_arm0 = None
        self.grasp_offset_arm1 = None
        
        # Track when grasp started (for holding penalty)
        self.grasp_start_step = None
        self.MAX_HOLD_STEPS = 200  # Penalty starts after this many steps holding

        # Get cube body ID
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cube')

        self.n_actuators = self.model.nu
        self.joint_limits_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.joint_limits_high = self.model.actuator_ctrlrange[:, 1].copy()
        
        # Define Spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_actuators,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(42,),
            dtype=np.float32
        )
        
        # Cache IDs
        self.cube_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "cube_site")
        self.arm0_ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "arm0_ee_site")
        self.arm1_ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "arm1_ee_site")
        
        # Get gripper geom IDs for contact detection
        self.gripper0_geom_ids = self._get_gripper_geom_ids("arm0")
        self.gripper1_geom_ids = self._get_gripper_geom_ids("arm1")

        self.viewer = None
        self._render_context = None
        
        # Init Poses
        self.base_init_pose = np.array([
            -0.38, 0.1, 0.05, -0.1, 0.0, 1.0,  # Arm 0 (gripper open)
            -0.38, 0.1, 0.05, -0.1, 0.0, 1.0   # Arm 1 (gripper open)
        ], dtype=np.float32)
        
        self.init_poses = [
            self.base_init_pose.copy(),
            self.base_init_pose + np.array([0.1, 0.02, 0.02, 0, 0, 0, -0.1, 0.02, 0.02, 0, 0, 0]),
            self.base_init_pose + np.array([-0.1, 0.03, 0.03, 0, 0, 0, 0.1, 0.03, 0.03, 0, 0, 0]),
        ]
        
        # Episode tracking
        self.episode_count = 0
        self.success_count = 0
        self.best_height = 0.19
        
        # Phase tracking
        # 0 = reaching, 1 = contact, 2 = grasping, 3 = lifting, 4 = placing
        self.current_phase = 0
        self.phase_entry_step = 0
        self.primary_arm = None
        
        # Track if cube was ever lifted (for success detection)
        self.cube_was_lifted = False
        self.max_cube_height = 0.19

    
    def _apply_sticky_gripper(self):
        """
        Apply sticky gripper effect with PROPER RELEASE.
        Key fix: Check if gripper is OPENING and release the cube.
        """
        cube_pos = self.data.site_xpos[self.cube_site_id].copy()
        
        # Check arm 0
        ee0_pos = self.data.site_xpos[self.arm0_ee_site_id].copy()
        dist0 = np.linalg.norm(ee0_pos - cube_pos)
        gripper0_state = self.data.qpos[5]
        gripper0_closed = gripper0_state < self.GRIPPER_CLOSED_THRESHOLD
        gripper0_open = gripper0_state > self.GRIPPER_OPEN_THRESHOLD  # NEW: Check if opening
        near0 = dist0 < self.GRASP_THRESHOLD
        
        # Check arm 1
        ee1_pos = self.data.site_xpos[self.arm1_ee_site_id].copy()
        dist1 = np.linalg.norm(ee1_pos - cube_pos)
        gripper1_state = self.data.qpos[11]
        gripper1_closed = gripper1_state < self.GRIPPER_CLOSED_THRESHOLD
        gripper1_open = gripper1_state > self.GRIPPER_OPEN_THRESHOLD  # NEW
        near1 = dist1 < self.GRASP_THRESHOLD
        
        # Handle grasp initiation for arm 0
        if near0 and gripper0_closed and not self.is_grasped_arm0:
            self.is_grasped_arm0 = True
            self.grasp_offset_arm0 = cube_pos - ee0_pos
            if self.grasp_start_step is None:
                self.grasp_start_step = self.current_step
        
        # Handle grasp initiation for arm 1
        if near1 and gripper1_closed and not self.is_grasped_arm1:
            self.is_grasped_arm1 = True
            self.grasp_offset_arm1 = cube_pos - ee1_pos
            if self.grasp_start_step is None:
                self.grasp_start_step = self.current_step
        
        # === KEY FIX: Release when gripper OPENS ===
        if gripper0_open and self.is_grasped_arm0:
            self.is_grasped_arm0 = False
            self.grasp_offset_arm0 = None
            # Clear grasp tracking if no arm is holding
            if not self.is_grasped_arm1:
                self.grasp_start_step = None
        
        if gripper1_open and self.is_grasped_arm1:
            self.is_grasped_arm1 = False
            self.grasp_offset_arm1 = None
            if not self.is_grasped_arm0:
                self.grasp_start_step = None
        
        # Apply magnetic force if still grasped
        if self.is_grasped_arm0 or self.is_grasped_arm1:
            self._apply_magnetic_force()
        else:
            # Clear forces when not grasped
            self.data.xfrc_applied[self.cube_body_id, :] = 0


    def _apply_magnetic_force(self):
        """Apply force pulling cube toward gripper with clamping."""
        cube_pos = self.data.site_xpos[self.cube_site_id].copy()
        total_force = np.zeros(3)
        
        if self.is_grasped_arm0 and self.grasp_offset_arm0 is not None:
            ee_pos = self.data.site_xpos[self.arm0_ee_site_id].copy()
            target_pos = ee_pos + self.grasp_offset_arm0
            displacement = target_pos - cube_pos
            total_force += displacement * self.STICKY_STRENGTH
        
        if self.is_grasped_arm1 and self.grasp_offset_arm1 is not None:
            ee_pos = self.data.site_xpos[self.arm1_ee_site_id].copy()
            target_pos = ee_pos + self.grasp_offset_arm1
            displacement = target_pos - cube_pos
            total_force += displacement * self.STICKY_STRENGTH
        
        # Clamp force
        force_magnitude = np.linalg.norm(total_force)
        if force_magnitude > self.MAX_FORCE:
            total_force = total_force / force_magnitude * self.MAX_FORCE
        
        self.data.xfrc_applied[self.cube_body_id, :3] = total_force
        
        # Damping
        cube_qvel_start = 12
        if cube_qvel_start < len(self.data.qvel) - 3:
            cube_vel = self.data.qvel[cube_qvel_start:cube_qvel_start+3]
            damping = -cube_vel * 2.0
            damping = np.clip(damping, -1.0, 1.0)
            self.data.xfrc_applied[self.cube_body_id, :3] += damping


    def _reset_sticky_gripper(self):
        """Reset sticky gripper state."""
        self.is_grasped_arm0 = False
        self.is_grasped_arm1 = False
        self.grasp_offset_arm0 = None
        self.grasp_offset_arm1 = None
        self.grasp_start_step = None
        if hasattr(self, 'cube_body_id') and self.cube_body_id >= 0:
            self.data.xfrc_applied[self.cube_body_id, :] = 0


    def _get_gripper_geom_ids(self, arm_prefix: str) -> list:
        """Get geom IDs for gripper parts."""
        geom_ids = []
        possible_names = [
            f"{arm_prefix}_gripper_left",
            f"{arm_prefix}_gripper_right", 
            f"{arm_prefix}_finger_left",
            f"{arm_prefix}_finger_right",
            f"{arm_prefix}_gripper",
            f"{arm_prefix}_fixed_pad",
            f"{arm_prefix}_moving_pad",
        ]
        for name in possible_names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                geom_ids.append(gid)
        return geom_ids
    
    def _check_gripper_contact(self, arm_idx: int) -> bool:
        """Check if gripper is in contact with cube."""
        gripper_geoms = self.gripper0_geom_ids if arm_idx == 0 else self.gripper1_geom_ids
        
        if not gripper_geoms or self.cube_body_id < 0:
            return False
            
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]
            
            is_cube_contact = (body1 == self.cube_body_id or body2 == self.cube_body_id)
            is_gripper_contact = (geom1 in gripper_geoms or geom2 in gripper_geoms)
            
            if is_cube_contact and is_gripper_contact:
                return True
        
        return False


    def _get_grasping_arm(self):
        """Return which arm is currently grasping (0, 1, or None)."""
        if self.is_grasped_arm0:
            return 0
        elif self.is_grasped_arm1:
            return 1
        return None

    
    def _get_obs(self) -> np.ndarray:
        """Get observation vector."""
        qpos = self.data.qpos[:12].copy()
        qvel = self.data.qvel[:12].copy()
        
        cube_pos = self.data.site_xpos[self.cube_site_id].copy()
        arm0_ee = self.data.site_xpos[self.arm0_ee_site_id].copy()
        arm1_ee = self.data.site_xpos[self.arm1_ee_site_id].copy()
        
        gripper0 = qpos[5]
        gripper1 = qpos[11]
        
        phase_normalized = self.current_phase / 4.0
        target_dist = np.linalg.norm(cube_pos[:2] - self.target_pos[:2])
        
        grasping_arm0 = 1.0 if self.is_grasped_arm0 else 0.0
        grasping_arm1 = 1.0 if self.is_grasped_arm1 else 0.0
        
        obs = np.concatenate([
            qpos,                    # 12
            qvel,                    # 12  
            cube_pos,                # 3
            arm0_ee,                 # 3
            arm1_ee,                 # 3
            [gripper0, gripper1],    # 2
            [phase_normalized],      # 1
            [target_dist],           # 1
            [grasping_arm0, grasping_arm1],  # 2
            self.target_pos,         # 3
        ]).astype(np.float32)
        
        return obs

    
    def _get_info(self) -> Dict[str, Any]:
        cube_pos = self.data.site_xpos[self.cube_site_id].copy()
        arm0_ee = self.data.site_xpos[self.arm0_ee_site_id].copy()
        arm1_ee = self.data.site_xpos[self.arm1_ee_site_id].copy()
        
        dist0 = np.linalg.norm(arm0_ee - cube_pos)
        dist1 = np.linalg.norm(arm1_ee - cube_pos)
        
        primary_arm = 0 if dist0 < dist1 else 1
        primary_dist = min(dist0, dist1)
        
        contact0 = self._check_gripper_contact(0) or dist0 < 0.04
        contact1 = self._check_gripper_contact(1) or dist1 < 0.04
        any_contact = contact0 or contact1
        
        cube_to_target = np.linalg.norm(cube_pos[:2] - self.target_pos[:2])
        cube_to_target_3d = np.linalg.norm(cube_pos - self.target_pos)
        
        TABLE_HEIGHT = 0.19
        grasping_arm = self._get_grasping_arm()
        
        # Success: cube at target, on table
        is_at_target = cube_to_target < 0.06
        is_on_table = TABLE_HEIGHT - 0.02 < cube_pos[2] < TABLE_HEIGHT + 0.03
        is_released = grasping_arm is None
        
        # Success if cube is at target and on table (released or not)
        is_success = is_at_target and is_on_table
        
        return {
            "cube_pos": cube_pos,
            "dist0": dist0,
            "dist1": dist1,
            "min_dist_to_cube": primary_dist,
            "cube_height": cube_pos[2],
            "is_success": is_success,
            "arm0_ee_pos": arm0_ee,
            "arm1_ee_pos": arm1_ee,
            "primary_arm": primary_arm,
            "contact0": contact0,
            "contact1": contact1,
            "any_contact": any_contact,
            "current_phase": self.current_phase,
            "cube_to_target": cube_to_target,
            "cube_to_target_3d": cube_to_target_3d,
            "grasping_arm": grasping_arm,
            "is_at_target": is_at_target,
            "is_on_table": is_on_table,
            "is_released": is_released,
        }


    def _update_phase(self, info: Dict) -> None:
        """Update task phase with proper placing detection."""
        cube_height = info["cube_height"]
        TABLE_HEIGHT = 0.19
        
        cube_to_target = info["cube_to_target"]
        grasping_arm = self._get_grasping_arm()
        
        # Track max height
        if cube_height > self.max_cube_height:
            self.max_cube_height = cube_height
            if cube_height > TABLE_HEIGHT + 0.02:
                self.cube_was_lifted = True
        
        # Phase transitions
        if self.current_phase == 0:  # REACHING
            if info["any_contact"] or info["min_dist_to_cube"] < 0.04:
                self.current_phase = 1
                self.phase_entry_step = self.current_step
                self.primary_arm = info["primary_arm"]
                    
        elif self.current_phase == 1:  # CONTACT
            if grasping_arm is not None:
                self.current_phase = 2
                self.phase_entry_step = self.current_step
                self.grasping_arm = grasping_arm
                    
        elif self.current_phase == 2:  # GRASPING
            if cube_height > TABLE_HEIGHT + 0.02:
                self.current_phase = 3
                self.phase_entry_step = self.current_step
                    
        elif self.current_phase == 3:  # LIFTING/TRANSPORTING
            # Go to placing when near target OR after enough time
            if cube_to_target < 0.15 or (self.current_step - self.phase_entry_step > 100):
                self.current_phase = 4
                self.phase_entry_step = self.current_step
        
        elif self.current_phase == 4:  # PLACING
            pass  # Stay in placing phase
        
        # Phase regression - lost the cube before placing
        if self.current_phase >= 2 and self.current_phase < 4:
            if grasping_arm is None and cube_height < TABLE_HEIGHT + 0.01:
                if cube_to_target > 0.1:  # Dropped but not at target
                    self.current_phase = 0
                    self.primary_arm = None
                    self.grasping_arm = None

    
    def _compute_reward(self, info: Dict[str, Any]) -> float:
        """
        Reward function with STRONG placing incentives.
        
        Key changes from V2:
        - Phase 4 has much stronger rewards for placing
        - Penalty for holding cube too long
        - Bonus for releasing at target
        """
        if self.reward_type == "sparse":
            return 100.0 if info["is_success"] else 0.0
        
        reward = 0.0
        TABLE_HEIGHT = 0.19
        
        cube_pos = info["cube_pos"]
        cube_height = info["cube_height"]
        cube_to_target = info["cube_to_target"]
        grasping_arm = self._get_grasping_arm()
        
        # === PHASE 0: REACHING ===
        if self.current_phase == 0:
            primary_dist = info["min_dist_to_cube"]
            reach_reward = 2.0 * (1.0 - np.tanh(5.0 * primary_dist))
            reward += reach_reward
        
        # === PHASE 1: CONTACT ===
        elif self.current_phase == 1:
            if info["any_contact"]:
                reward += 1.0
            
            if self.primary_arm is not None:
                gripper_idx = 5 if self.primary_arm == 0 else 11
                gripper_val = self.data.qpos[gripper_idx]
                reward += (1.0 - gripper_val) * 1.5
            
            if grasping_arm is not None:
                reward += 3.0
        
        # === PHASE 2: GRASPING ===
        elif self.current_phase == 2:
            if grasping_arm is not None:
                reward += 1.5
                if cube_height > TABLE_HEIGHT + 0.01:
                    reward += 2.0
            else:
                reward -= 2.0
        
        # === PHASE 3: LIFTING & TRANSPORTING ===
        elif self.current_phase == 3:
            # Reward for height (but cap it - don't want cube in space)
            height_above = cube_height - TABLE_HEIGHT
            if height_above > 0:
                # Sweet spot: 3-8cm above table
                if 0.03 < height_above < 0.08:
                    reward += 2.0
                elif height_above > 0.15:
                    # Too high - encourage lowering
                    reward -= 1.0
                else:
                    reward += height_above * 10.0
            
            # Strong reward for moving toward target
            transport_reward = 4.0 * (1.0 - np.tanh(3.0 * cube_to_target))
            reward += transport_reward
            
            # Maintain grasp during transport
            if grasping_arm is not None:
                reward += 0.5
            else:
                reward -= 3.0
        
        # === PHASE 4: PLACING (KEY PHASE) ===
        elif self.current_phase == 4:
            # Strong reward for being at target location
            at_target_reward = 5.0 * (1.0 - np.tanh(5.0 * cube_to_target))
            reward += at_target_reward
            
            if cube_to_target < 0.1:
                reward += 3.0  # Close to target bonus
                
                # Reward for lowering cube toward table
                if grasping_arm is not None:
                    # Still holding - encourage lowering
                    target_height = TABLE_HEIGHT + 0.02
                    height_error = abs(cube_height - target_height)
                    reward += 2.0 * (1.0 - np.tanh(10.0 * height_error))
                    
                    # ENCOURAGE OPENING GRIPPER when low enough
                    if cube_height < TABLE_HEIGHT + 0.05:
                        gripper_idx = 5 if grasping_arm == 0 else 11
                        gripper_val = self.data.qpos[gripper_idx]
                        # Reward gripper opening (high value = open)
                        reward += gripper_val * 3.0
                
            # === RELEASE BONUS ===
            if cube_to_target < 0.08:
                if grasping_arm is None:  # Released!
                    if cube_height < TABLE_HEIGHT + 0.04:
                        # Successfully placed!
                        reward += 20.0
                    else:
                        # Released but cube is floating? (shouldn't happen)
                        reward += 5.0
            
            # Cube on table at target = BIG BONUS
            if info["is_on_table"] and info["is_at_target"]:
                reward += 30.0
        
        # === HOLDING PENALTY (encourages release) ===
        if self.grasp_start_step is not None and grasping_arm is not None:
            hold_duration = self.current_step - self.grasp_start_step
            if hold_duration > self.MAX_HOLD_STEPS:
                # Progressive penalty for holding too long
                overtime = hold_duration - self.MAX_HOLD_STEPS
                reward -= 0.02 * overtime
        
        # === SINGLE-ARM ENFORCEMENT ===
        if grasping_arm is not None:
            other_arm = 1 - grasping_arm
            other_dist = info["dist1"] if grasping_arm == 0 else info["dist0"]
            other_ee = info["arm1_ee_pos"] if grasping_arm == 0 else info["arm0_ee_pos"]
            
            if other_dist < 0.12:
                reward -= 2.0 * (0.12 - other_dist) / 0.12
            
            safe_pos = self.arm1_safe_pos if grasping_arm == 0 else self.arm0_safe_pos
            dist_to_safe = np.linalg.norm(other_ee - safe_pos)
            if dist_to_safe < 0.15:
                reward += 0.5
        
        # === UNIVERSAL REWARDS/PENALTIES ===
        
        # SUCCESS!
        if info["is_success"]:
            reward += 100.0
        
        # Arm collision penalty
        ee_dist = np.linalg.norm(info["arm0_ee_pos"] - info["arm1_ee_pos"])
        if ee_dist < 0.06:
            reward -= 3.0
        
        # Cube knocked off table
        if cube_height < 0.1:
            reward -= 20.0
        
        # Cube too far from workspace
        if np.linalg.norm(cube_pos[:2]) > 0.25:
            reward -= 5.0
        
        # Energy penalty
        reward -= 0.002 * np.linalg.norm(self.data.ctrl)
        
        return reward


    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_count += 1
        self.current_phase = 0
        self.phase_entry_step = 0
        self.primary_arm = None
        self.grasping_arm = None
        self._reset_sticky_gripper()
        self.cube_was_lifted = False
        self.max_cube_height = 0.19
    
        mujoco.mj_resetData(self.model, self.data)
        
        # Set Robot Pose
        if self.use_multi_init and self.np_random is not None:
            idx = self.np_random.integers(0, len(self.init_poses))
            init_pose = self.init_poses[idx].copy()
            noise = self.np_random.uniform(-0.05, 0.05, size=12)
            init_pose += noise
        else:
            init_pose = self.base_init_pose.copy()
            
        # Grippers start open
        init_pose[5] = 1.0
        init_pose[11] = 1.0
        
        self.data.qpos[:12] = init_pose
        self.data.ctrl[:] = init_pose
        
        # Set Cube Pose
        TABLE_HEIGHT = 0.19
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        if cube_joint_id >= 0:
            adr = self.model.jnt_qposadr[cube_joint_id]
            self.data.qpos[adr] = np.random.uniform(-0.03, 0.03)
            self.data.qpos[adr+1] = np.random.uniform(-0.03, 0.03)
            self.data.qpos[adr+2] = TABLE_HEIGHT
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        self.current_step += 1
        action = np.clip(action, -1.0, 1.0)
        
        current_ctrl = self.data.ctrl.copy()
        target_ctrl = current_ctrl + (action * self.action_scale)
        target_ctrl = np.clip(target_ctrl, self.joint_limits_low, self.joint_limits_high)
        self.data.ctrl[:] = target_ctrl
        
        self._apply_sticky_gripper()
        
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
            
        obs = self._get_obs()
        info = self._get_info()
        
        self._update_phase(info)
        info["current_phase"] = self.current_phase
        
        reward = self._compute_reward(info)
        
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

    def print_stats(self):
        rate = 100 * self.success_count / max(1, self.episode_count)
        print(f"Ep: {self.episode_count}, Success: {self.success_count} ({rate:.1f}%)")


if __name__ == "__main__":
    print("Testing DualSO101PickCubeEnv V3...")
    env = DualSO101PickCubeEnv(render_mode=None)
    obs, info = env.reset(seed=42)
    print(f"Obs shape: {obs.shape}")
    print(f"Initial phase: {info['current_phase']}")
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        if i % 20 == 0:
            print(f"Step {i}: phase={info['current_phase']}, reward={reward:.2f}, "
                  f"grasped={env._get_grasping_arm()}, cube_h={info['cube_height']:.3f}")
    
    env.close()
    print("Test complete!")