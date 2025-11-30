import numpy as np
import matplotlib.pyplot as plt
from solver import StackedResolvedRateControl

def run_experiment(mode="standard"):
    # Initialize Solver
    ctrl = StackedResolvedRateControl()
    
    # Simulation Parameters (Matching visualize.py)
    dt = 0.05
    steps = 200
    
    # Initial State (Bent elbows)
    q_current = np.zeros(10)
    q_current[2] = 0.5 
    q_current[7] = 0.5 
    
    # --- CALCULATE START POS ---
    q_A_start = q_current[0:5]
    q_B_start = q_current[5:10]
    T_start = ctrl.get_forward_kinematics_stacked(q_A_start, q_B_start)
    start_pos = T_start.t 
    
    start_y = start_pos[1]
    start_z = start_pos[2]
    
    # Circle Parameters
    radius = 0.1
    speed = 0.8  # The tuned speed
    Kp = 30.0    # The tuned Gain
    
    center_y = start_y
    center_z = start_z + radius 

    # DATA STORAGE
    velocity_history = []
    error_history = []
    
    print(f"Running {mode} circle experiment...")
    
    for i in range(steps):
        t = i * dt
        
        # --- 1. TARGETS ---
        angle = (speed * t) - (np.pi / 2)
        target_y = center_y + radius * np.cos(angle)
        target_z = center_z + radius * np.sin(angle)
        
        vy_ff = -radius * speed * np.sin(angle)
        vz_ff =  radius * speed * np.cos(angle)
        
        # --- 2. ACTUALS ---
        q_A_now = q_current[0:5]
        q_B_now = q_current[5:10]
        T_current = ctrl.get_forward_kinematics_stacked(q_A_now, q_B_now)
        current_pos = T_current.t
        
        # --- 3. ERROR ---
        error_y = target_y - current_pos[1]
        error_z = target_z - current_pos[2]
        
        # Store Error Magnitude (mm)
        error_mag = np.sqrt(error_y**2 + error_z**2) * 1000
        error_history.append(error_mag)
        
        # --- 4. CONTROL ---
        vy_cmd = vy_ff + (Kp * error_y)
        vz_cmd = vz_ff + (Kp * error_z)
        target_vel_corrected = np.array([0.0, vy_cmd, vz_cmd, 0, 0, 0])

        # --- 5. WEIGHTS ---
        weights = np.ones(10)
        if mode == "weighted":
            # The Tuned Weight from your simulation
            weights[0:5] = 2.5 
        
        # Solve
        q_dot = ctrl.solve_velocity(q_current, target_vel_corrected, weights)
        
        # Store Data (deg/s)
        velocity_history.append(np.degrees(q_dot))
        
        # Integrate
        q_current = q_current + (q_dot * dt)
        
    return np.array(velocity_history), np.array(error_history)

def plot_results(vel_std, err_std, vel_w, err_w):
    time_axis = np.arange(len(vel_std)) * 0.05
    
    # Create 3 subplots now
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # --- PLOT 1: BOTTOM ARM EFFORT ---
    bottom_effort_std = np.sum(np.abs(vel_std[:, 0:5]), axis=1)
    bottom_effort_w = np.sum(np.abs(vel_w[:, 0:5]), axis=1)
    
    ax1.plot(time_axis, bottom_effort_std, 'r--', label='Standard (Equal)')
    ax1.plot(time_axis, bottom_effort_w, 'g-', linewidth=2, label='Weighted (Heavy Base)')
    ax1.set_title("1. Conservation of Energy: Bottom Arm Velocity")
    ax1.set_ylabel("Speed (deg/s)")
    ax1.legend()
    ax1.grid(True)
    
    # --- PLOT 2: TOP ARM EFFORT ---
    top_effort_std = np.sum(np.abs(vel_std[:, 5:10]), axis=1)
    top_effort_w = np.sum(np.abs(vel_w[:, 5:10]), axis=1)
    
    ax2.plot(time_axis, top_effort_std, 'r--', label='Standard')
    ax2.plot(time_axis, top_effort_w, 'g-', linewidth=2, label='Weighted')
    ax2.set_title("2. Load Transfer: Top Arm Velocity")
    ax2.set_ylabel("Speed (deg/s)")
    ax2.legend()
    ax2.grid(True)

    # --- PLOT 3: ACCURACY (ERROR) ---
    ax3.plot(time_axis, err_std, 'r--', label='Standard Error')
    ax3.plot(time_axis, err_w, 'g-', linewidth=2, label='Weighted Error')
    ax3.set_title("3. Accuracy Trade-off: Tracking Error")
    ax3.set_ylabel("Error (mm)")
    ax3.set_xlabel("Time (s)")
    ax3.legend()
    ax3.grid(True)
    # Zoom in on Y-axis to see small errors clearly
    ax3.set_ylim(0, 5.0) 
    
    print("Saving plot to 'circle_comparison.png'...")
    plt.savefig("circle_comparison.png")
    plt.show()

if __name__ == "__main__":
    # 1. Run Standard
    v_std, e_std = run_experiment("standard")
    
    # 2. Run Weighted
    v_w, e_w = run_experiment("weighted")
    
    # 3. Plot
    plot_results(v_std, e_std, v_w, e_w)