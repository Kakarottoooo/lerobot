import numpy as np
import roboticstoolbox as rtb
import matplotlib.animation as animation
from solver import StackedResolvedRateControl

def simulate():
    # 1. Initialize Math
    ctrl = StackedResolvedRateControl()
    
    print("Launching Visualizer with Perfect Alignment...")
    env = rtb.backends.PyPlot.PyPlot()
    env.launch()
    
    # 2. Add Robots
    ctrl.bot_A.q = np.zeros(6) 
    ctrl.bot_B.q = np.zeros(6)
    env.add(ctrl.bot_A)
    env.add(ctrl.bot_B)

    # 3. Camera Side View
    env.ax.view_init(elev=0, azim=0)

    # 4. Simulation Config
    dt = 0.05
    steps = 200 
    
    # Initial State (Bent elbows to avoid startup singularity)
    q_current = np.zeros(10)
    q_current[2] = 0.5 
    q_current[7] = 0.5 
    
    # --- FIX 1: CALCULATE START POS CORRECTLY ---
    q_A_start = q_current[0:5]
    q_B_start = q_current[5:10]
    T_start = ctrl.get_forward_kinematics_stacked(q_A_start, q_B_start)
    start_pos = T_start.t 
    
    # Start the circle exactly where the robot IS right now
    start_y = start_pos[1]
    start_z = start_pos[2]
    
    # Circle Parameters
    radius = 0.1
    speed = 1.0 # Slower speed for better tracking
    
    # Center is defined relative to the START point
    # We want the robot to start at the "Bottom" of the circle ( -pi/2 )
    center_y = start_y
    center_z = start_z + radius 

    # Trail for visualization
    trail_y = []
    trail_z = []
    line, = env.ax.plot([], [], [], 'r-', linewidth=2)

    print("\n--- DATA LOG ---")
    print(f"{'Step':<5} | {'Error (mm)':<10} | {'Status'}")
    print("-" * 35)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Ziwei Guo'), bitrate=2000)
    output_filename = "simulation_video_circle.mp4"
    
    Kp = 8.0 # Stronger error correction
    
    with writer.saving(env.fig, output_filename, dpi=100):
        for i in range(steps):
            t = i * dt
            
            # --- TARGET CALCULATION (Offset by -pi/2 so we start at 0) ---
            # Angle moves from -90 degrees to 270 degrees
            angle = (speed * t) - (np.pi / 2)
            
            target_y = center_y + radius * np.cos(angle)
            target_z = center_z + radius * np.sin(angle)
            
            # Velocity Feedforward
            vy_ff = -radius * speed * np.sin(angle)
            vz_ff =  radius * speed * np.cos(angle)
            
            # --- CURRENT STATE ---
            q_A_now = q_current[0:5]
            q_B_now = q_current[5:10]
            T_current = ctrl.get_forward_kinematics_stacked(q_A_now, q_B_now)
            current_pos = T_current.t
            
            # --- ERROR CALCULATION ---
            error_y = target_y - current_pos[1]
            error_z = target_z - current_pos[2]
            error_mm = np.sqrt(error_y**2 + error_z**2) * 1000
            
            # --- LOGGING ---
            trail_y.append(current_pos[1])
            trail_z.append(current_pos[2])
            
            # Update Red Line
            trail_x = [start_pos[0]] * len(trail_y) 
            line.set_data(trail_x, trail_y)
            line.set_3d_properties(trail_z)

            if i % 10 == 0:
                status = "PERFECT" if error_mm < 2.0 else "DRIFTING"
                print(f"{i:<5} | {error_mm:.3f} mm   | {status}")

            # --- CONTROL ---
            vy_cmd = vy_ff + (Kp * error_y)
            vz_cmd = vz_ff + (Kp * error_z)
            target_vel_corrected = np.array([0.0, vy_cmd, vz_cmd, 0, 0, 0])

            # --- FIX 2: RELAX WEIGHTS ---
            # 5.0 is enough to prove the point without breaking the robot
            weights = np.ones(10)
            weights[0:5] = 5.0 
            
            q_dot = ctrl.solve_velocity(q_current, target_vel_corrected, weights)
            q_current = q_current + (q_dot * dt)
            
            # --- UPDATE VIZ ---
            q_A_viz = np.append(q_current[0:5], 0.0)
            q_B_viz = np.append(q_current[5:10], 0.0)
            
            ctrl.bot_A.q = q_A_viz
            ctrl.bot_B.q = q_B_viz
            T_A_tip = ctrl.bot_A.fkine(q_A_viz)
            ctrl.bot_B.base = T_A_tip
            
            env.step()
            writer.grab_frame()
        
    print(f"Done. Video saved as '{output_filename}'")

if __name__ == "__main__":
    simulate()