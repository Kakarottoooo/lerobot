import numpy as np
import roboticstoolbox as rtb
import os

class StackedResolvedRateControl:
    def __init__(self, urdf_filename="so101.urdf"):
        # --- 1. ROBUST FILE LOADING ---
        # Get the folder where THIS python script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Join it with the urdf filename
        urdf_path = os.path.join(script_dir, urdf_filename)
        
        print(f"Loading robot from: {urdf_path}")

        try:
            # Load the same robot model twice (one for bottom, one for top)
            self.bot_A = rtb.ERobot.URDF(urdf_path)
            self.bot_B = rtb.ERobot.URDF(urdf_path)
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load URDF at {urdf_path}")
            print(f"Error details: {e}")
            print("Make sure 'so101.urdf' is in the same folder as this script!")
            return

        # --- 2. DOF CONFIGURATION ---
        # The So-101 URDF has 6 joints (Waist, Shoulder, Elbow, Wrist Flex, Wrist Roll, Gripper).
        # We only want to control the first 5 for the arm motion.
        # The 6th joint (Gripper) does not affect the arm's Jacobian.
        
        self.n_dof_single = 5 
        self.total_dof = self.n_dof_single * 2 # 10 DOF total for the stacked system
        
        print(f"System initialized. Controlling {self.total_dof} joints ({self.n_dof_single} per arm).")

    def get_forward_kinematics_stacked(self, q_A, q_B):
        """
        Calculates the final position of the stacked robot (Tip of B).
        """
        # FK of Robot A (Base -> Tip A)
        T_A = self.bot_A.fkine(q_A)
        
        # FK of Robot B (Tip A -> Tip B)
        # We treat Tip A as the "Base" for Robot B
        T_B = self.bot_B.fkine(q_B)
        
        # Combine: T_total = T_A * T_B
        T_total = T_A * T_B
        return T_total

    def get_combined_jacobian(self, q_A, q_B):
        """
        Calculates the Jacobian for the stacked system.
        Dimensions: 6 rows (vx,vy,vz,wx,wy,wz) x 10 columns (joints).
        """
        # --- PREPARE DATA ---
        # Ensure we only use the first 5 joints (ignore gripper)
        # q_A and q_B should be length 5 arrays coming in, or we slice them.
        
        # 1. Jacobian of Robot A (in World Frame)
        # This describes velocity at Tip A.
        J_A_local = self.bot_A.jacob0(q_A) # Returns 6x6 usually
        J_A_local = J_A_local[:, :self.n_dof_single] # Slice to 6x5

        # 2. Jacobian of Robot B (in Robot B's Base Frame)
        J_B_local = self.bot_B.jacob0(q_B)
        J_B_local = J_B_local[:, :self.n_dof_single] # Slice to 6x5

        # --- THE CORE MATH (Adjoint Transform) ---
        
        # Calculate transform from World (Base A) to Base B (Tip A)
        T_A = self.bot_A.fkine(q_A)
        
        # Calculate transform from Base B to Tip B
        T_B = self.bot_B.fkine(q_B)
        
        # A. Transform J_B into the World Frame.
        # The orientation of Base B is determined by T_A.
        # Velocity in World = Rotation_A * Velocity_in_B_frame
        R_A = T_A.R # Rotation matrix of A (3x3)
        zeros = np.zeros((3,3))
        # Create Block Diagonal Matrix for rotation
        rot_block = np.block([[R_A, zeros], [zeros, R_A]])
        
        J_B_world = rot_block @ J_B_local

        # B. Transform J_A to act on the final End Effector (Tip B).
        # A moves B's base. The "Lever Arm" effect means angular velocity at A 
        # creates linear velocity at Tip B.
        
        # Vector from Tip A to Tip B (in B's frame)
        p_B_tip_relative = T_B.t 
        # Convert this vector to world frame
        p_lever = R_A @ p_B_tip_relative
        
        # Skew symmetric matrix of lever arm (for cross product)
        lx, ly, lz = p_lever
        skew_L = np.array([
            [0, -lz, ly],
            [lz, 0, -lx],
            [-ly, lx, 0]
        ])
        
        # J_A_linear_new = J_A_linear_old - (skew_L @ J_A_angular)
        J_A_linear = J_A_local[0:3, :]
        J_A_angular = J_A_local[3:6, :]
        
        J_A_linear_at_tip = J_A_linear - (skew_L @ J_A_angular)
        
        # Stack them back together
        J_A_effective = np.vstack((J_A_linear_at_tip, J_A_angular))

        # --- COMBINE ---
        # J_total = [ J_A_effective, J_B_world ]
        J_total = np.hstack((J_A_effective, J_B_world))
        
        return J_total

    def weighted_pseudo_inverse(self, J, W):
        """
        Implements: q_dot = W^-1 * J.T * (J * W^-1 * J.T)^-1 * x_dot
        """
        # Calculate inverse of W (Diagonal matrix)
        W_inv = np.linalg.inv(W)
        
        # Damped Least Squares for numerical stability (prevents crash at singularity)
        damping = 1e-4 * np.eye(6)
        
        # The Math:
        term1 = J @ W_inv @ J.T
        term1_inv = np.linalg.inv(term1 + damping)
        
        J_w_pinv = W_inv @ J.T @ term1_inv
        return J_w_pinv

    def solve_velocity(self, q_full, desired_velocity, weights):
        """
        Main solver function.
        Args:
            q_full: 10-element array (Angles for all joints)
            desired_velocity: 6-element array (vx, vy, vz, wx, wy, wz)
            weights: 10-element array (Cost of moving each joint)
        Returns:
            q_dot: 10-element array (Velocities for all joints)
        """
        # Split q into A and B
        n = self.n_dof_single
        q_A = q_full[0:n]
        q_B = q_full[n:2*n]
        
        # Get Jacobian
        J = self.get_combined_jacobian(q_A, q_B)
        
        # Weight Matrix
        W = np.diag(weights)
        
        # Solve
        J_pinv = self.weighted_pseudo_inverse(J, W)
        q_dot = J_pinv @ desired_velocity
        
        return q_dot

# --- Test Block ---
if __name__ == "__main__":
    # Initialize logic
    solver = StackedResolvedRateControl(urdf_filename="so101.urdf")
    
    # Check if robot loaded successfully
    if hasattr(solver, 'bot_A'):
        # 1. Define Current State (10 Joints, all at 0 radians)
        q_test = np.zeros(10) 
        
        # 2. Define Command: Move Straight UP in Z axis
        # [vx, vy, vz, wx, wy, wz]
        x_dot = np.array([0, 0, 0.1, 0, 0, 0])
        
        # 3. Define Weights (1.0 = Standard)
        # Try changing these! If you put 100.0 on the first 5, the bottom arm will stop moving.
        weights = np.ones(10)
        
        # 4. Calculate
        res = solver.solve_velocity(q_test, x_dot, weights)
        
        print("\n--- RESULT ---")
        print(f"Input Velocity: {x_dot}")
        print("Calculated Joint Velocities (q_dot):")
        with np.printoptions(precision=4, suppress=True):
            print(res)