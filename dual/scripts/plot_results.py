import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ================= CONFIGURATION =================
LOG_DIR = "dual/logs"  # The directory defined in your train scripts
OUTPUT_FILE = "learning_curves.png"
SMOOTHING = 0.9  # Smoothing factor (0.0 to 0.99)
# =================================================

def smooth(scalars, weight):
    """
    EMA implementation for smoothing curves like TensorBoard does.
    """
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def parse_tensorboard(path):
    """
    Parses a TensorBoard tfevents file and returns a DataFrame.
    """
    ea = EventAccumulator(
        path,
        size_guidance={
            EventAccumulator.COMPRESSED_HISTOGRAMS: 500,
            EventAccumulator.IMAGES: 4,
            EventAccumulator.AUDIO: 4,
            EventAccumulator.SCALARS: 0,
            EventAccumulator.HISTOGRAMS: 1,
        },
    )
    ea.Reload()

    # defined tags in your script
    tags = [
        'rollout/ep_rew_mean',  # Standard SB3 reward
        'custom/success_rate'   # Your custom callback
    ]
    
    data = {}
    
    for tag in tags:
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [x.step for x in events]
            values = [x.value for x in events]
            
            # Apply smoothing
            if len(values) > 0:
                values = smooth(values, SMOOTHING)
                
            data[tag] = pd.DataFrame({'step': steps, 'value': values})
            
    return data

def main():
    print(f"Scanning directory: {LOG_DIR}...")
    
    # Find all event files
    # Structure is usually: dual/logs/sac_v12_DATE/events.out.tfevents...
    files = glob.glob(f"{LOG_DIR}/**/*tfevents*", recursive=True)
    
    if not files:
        print("No TensorBoard files found! Check your LOG_DIR.")
        return

    dfs_reward = []
    dfs_success = []

    print(f"Found {len(files)} log files. Parsing...")

    for f in files:
        # Determine Algorithm (SAC or PPO) based on folder name
        parent_folder = os.path.dirname(f)
        algo = "Unknown"
        if "sac" in parent_folder.lower():
            algo = "SAC"
        elif "ppo" in parent_folder.lower():
            algo = "PPO"
        
        print(f"Processing {algo} run: {parent_folder}")
        
        try:
            data = parse_tensorboard(f)
            
            # Process Reward
            if 'rollout/ep_rew_mean' in data:
                df = data['rollout/ep_rew_mean']
                df['Algorithm'] = algo
                dfs_reward.append(df)

            # Process Success Rate
            if 'custom/success_rate' in data:
                df = data['custom/success_rate']
                df['Algorithm'] = algo
                dfs_success.append(df)
                
        except Exception as e:
            print(f"Error parsing {f}: {e}")

    # Combine Data
    if not dfs_reward or not dfs_success:
        print("Could not extract enough data to plot.")
        return

    full_df_reward = pd.concat(dfs_reward, ignore_index=True)
    full_df_success = pd.concat(dfs_success, ignore_index=True)

    # ================= PLOTTING =================
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot 1: Mean Episode Reward
    sns.lineplot(
        data=full_df_reward, 
        x="step", 
        y="value", 
        hue="Algorithm", 
        ax=axes[0],
        linewidth=2,
        palette=["#1f77b4", "#ff7f0e"] # Blue for SAC, Orange for PPO
    )
    axes[0].set_ylabel("Mean Episode Reward")
    axes[0].set_title("Training Progress: Reward")
    axes[0].legend(loc="upper left")

    # Plot 2: Success Rate
    sns.lineplot(
        data=full_df_success, 
        x="step", 
        y="value", 
        hue="Algorithm", 
        ax=axes[1],
        linewidth=2,
        palette=["#1f77b4", "#ff7f0e"]
    )
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_xlabel("Timesteps")
    axes[1].set_title("Training Progress: Success Rate")
    axes[1].set_ylim(0, 105) # 0 to 100%
    
    # Add 50% success line
    axes[1].axhline(50, color='gray', linestyle='--', alpha=0.5, label="50% Success")

    

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"\nSuccess! Plot saved to {OUTPUT_FILE}")
    plt.show()

if __name__ == "__main__":
    main()