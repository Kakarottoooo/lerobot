import numpy as np
import roboticstoolbox as rtb
import time
import matplotlib.animation as animation # <--- New Import
from solver import StackedResolvedRateControl

def simulate():
    # 1. Initialize Math
    ctrl = StackedResolvedRateControl()
    
    # 2. Launch Matplotlib Visualizer
    print("Launching Matplotlib Visualizer...")
    env = rtb.backends.PyPlot.PyPlot()
    env.launch()
    
    # 3. Add Robots to Scene
    ctrl.bot_A.q = np.zeros(6) 
    ctrl.bot_B.q = np.zeros(6)
    
    env.add(ctrl.bot_A)
    env.add(ctrl.bot_B)

    # 4. Simulation Config
    dt = 0.05
    steps = 100 # Total frames
    
    # State vectors
    q_current = np.zeros(10)
    target_vel = np.array([0.0, 0.02, 0.05, 0, 0, 0])
    
    print("Starting Simulation...")

    # ### VIDEO RECORDING SETUP ###
    # We define a writer object that grabs frames from the plot
    # fps=20 means the video will play at 20 frames per second
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Ziwei Guo'), bitrate=1800)

    # We wrap the simulation loop inside the writer 'saving' context
    # 'env.fig' is the Matplotlib figure window created by the toolbox
    output_filename = "simulation_video_2.mp4"
    print(f"Recording video to {output_filename}...")
    
    with writer.saving(env.fig, output_filename, dpi=100):
        for i in range(steps):
            # --- A. MATH STEP ---
            weights = np.ones(10)
            q_dot = ctrl.solve_velocity(q_current, target_vel, weights)
            
            # Integrate
            q_current = q_current + (q_dot * dt)
            
            # --- B. VISUALIZATION STEP ---
            q_A_math = q_current[0:5]
            q_B_math = q_current[5:10]
            
            # Append 0 for gripper
            q_A_viz = np.append(q_A_math, 0.0)
            q_B_viz = np.append(q_B_math, 0.0)
            
            # Update robots
            ctrl.bot_A.q = q_A_viz
            ctrl.bot_B.q = q_B_viz
            
            # --- C. STACKING ---
            T_A_tip = ctrl.bot_A.fkine(q_A_viz)
            ctrl.bot_B.base = T_A_tip
            
            # Update Plot
            env.step()
            
            # ### CAPTURE FRAME ###
            # Instead of sleeping, we grab the frame
            writer.grab_frame()
            
            # Print progress every 10 frames
            if i % 10 == 0:
                print(f"Rendering frame {i}/{steps}")
        
    print(f"Done. Video saved as '{output_filename}'")
    # Keep window open at the end if you want to see the final pose
    # env.hold() 

if __name__ == "__main__":
    simulate()