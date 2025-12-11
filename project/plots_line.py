import numpy as np
import matplotlib.pyplot as plt
from solver import StackedResolvedRateControl

def run_experiment(mode="standard"):
    # Initialize Solver
    ctrl = StackedResolvedRateControl()
    
    # Simulation Parameters
    dt = 0.05
    steps = 100
    q_current = np.zeros(10)
    
    # Target: Move UP (Z+) and Sideways (Y+)
    target_vel = np.array([0.0, 0.05, 0.05, 0, 0, 0])
    
    # DATA STORAGE
    velocity_history = []
    
    # Define Weights based on mode
    weights = np.ones(10)
    if mode == "weighted":
        # Make the Bottom Arm (Joints 0-5) expensive to move (Weight = 10)
        # Keep Top Arm (Joints 5-10) cheap to move (Weight = 1)
        weights[0:5] = 10.0 
    
    print(f"Running {mode} experiment...")
    
    for i in range(steps):
        # Solve
        q_dot = ctrl.solve_velocity(q_current, target_vel, weights)
        
        # Store data (convert to degrees/sec for easier reading)
        velocity_history.append(np.degrees(q_dot))
        
        # Integrate
        q_current = q_current + (q_dot * dt)
        
    return np.array(velocity_history)

def plot_results(data_std, data_w):
    time_axis = np.arange(len(data_std)) * 0.05
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # --- PLOT 1: BOTTOM ARM VELOCITIES ---
    # We sum the absolute velocity of the bottom arm to see total "effort"
    # Bottom Arm indices: 0, 1, 2, 3, 4
    bottom_effort_std = np.sum(np.abs(data_std[:, 0:5]), axis=1)
    bottom_effort_w = np.sum(np.abs(data_w[:, 0:5]), axis=1)
    
    ax1.plot(time_axis, bottom_effort_std, 'r--', label='Standard (Equal Weights)')
    ax1.plot(time_axis, bottom_effort_w, 'g-', linewidth=2, label='Weighted (Heavy Base)')
    ax1.set_title("Comparison: Total Velocity of Bottom Arm")
    ax1.set_ylabel("Total Joint Speed (deg/s)")
    ax1.legend()
    ax1.grid(True)
    
    # --- PLOT 2: TOP ARM VELOCITIES ---
    # Top Arm indices: 5, 6, 7, 8, 9
    top_effort_std = np.sum(np.abs(data_std[:, 5:10]), axis=1)
    top_effort_w = np.sum(np.abs(data_w[:, 5:10]), axis=1)
    
    ax2.plot(time_axis, top_effort_std, 'r--', label='Standard')
    ax2.plot(time_axis, top_effort_w, 'g-', linewidth=2, label='Weighted')
    ax2.set_title("Comparison: Total Velocity of Top Arm")
    ax2.set_ylabel("Total Joint Speed (deg/s)")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    ax2.grid(True)
    
    print("Saving plot to 'comparison_results.png'...")
    plt.savefig("comparison_results.png")
    plt.show()

if __name__ == "__main__":
    # 1. Run Standard
    data_standard = run_experiment("standard")
    
    # 2. Run Weighted
    data_weighted = run_experiment("weighted")
    
    # 3. Plot
    plot_results(data_standard, data_weighted)