"""
PPO Training Script for Dual-Arm SO-101 (V12 Compatible)

Key Features:
- Includes Windows Multiprocessing Fix
- Uses VecNormalize (Critical for PPO)
- Includes Curriculum and Success Rate logging
- Includes Evaluation Mode (--eval)
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

# Add project root for the main process
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback, 
    BaseCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor


def make_env(rank, seed=0, use_curriculum=True):
    """Create a single environment instance"""
    def _init():
        # --- WINDOWS MULTIPROCESSING FIX ---
        import sys
        import os
        # Force add project root to path for the child process
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
        # -----------------------------------

        # Correct import from your actual filename
        from dual.envs.dual_so101_env import DualSO101PickCubeEnv
        
        env = DualSO101PickCubeEnv(
            render_mode=None,
            max_episode_steps=500,
            use_curriculum=use_curriculum,
            curriculum_progress=0.0,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


class CurriculumCallback(BaseCallback):
    """Update curriculum progress based on training progress"""
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        
    def _on_step(self):
        progress = self.num_timesteps / self.total_timesteps
        vec_env = self.training_env
        
        # Unwrap VecNormalize/VecEnv wrappers to find the base env
        if hasattr(vec_env, 'venv'):
            vec_env = vec_env.venv
        if hasattr(vec_env, 'envs'):
            for env in vec_env.envs:
                base_env = env
                while hasattr(base_env, 'env'):
                    base_env = base_env.env
                if hasattr(base_env, 'curriculum_progress'):
                    base_env.curriculum_progress = progress
        return True


class SuccessRateCallback(BaseCallback):
    """Track and log success rates"""
    def __init__(self, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_successes = []
        self.episode_grasps = []
        self.episode_lifts = []
        
    def _on_step(self):
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_successes.append(info.get('is_success', False))
                    self.episode_grasps.append(info.get('was_grasped', False))
                    self.episode_lifts.append(info.get('was_lifted', False))
        
        if self.num_timesteps % self.check_freq == 0 and len(self.episode_successes) > 0:
            n = len(self.episode_successes)
            success_rate = sum(self.episode_successes) / n * 100
            grasp_rate = sum(self.episode_grasps) / n * 100
            lift_rate = sum(self.episode_lifts) / n * 100
            
            print(f"\n[PPO Steps: {self.num_timesteps}] Episodes: {n}")
            print(f"  Grasp: {grasp_rate:.1f}% | Lift: {lift_rate:.1f}% | Success: {success_rate:.1f}%")
            
            self.logger.record("custom/success_rate", success_rate)
            self.logger.record("custom/grasp_rate", grasp_rate)
            self.logger.record("custom/lift_rate", lift_rate)
            
            self.episode_successes = []
            self.episode_grasps = []
            self.episode_lifts = []
        return True


class SaveNormCallback(BaseCallback):
    """Save VecNormalize stats with checkpoints"""
    def __init__(self, save_path, save_freq=50000, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        
    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0:
            norm_path = os.path.join(self.save_path, f"vecnormalize_{self.num_timesteps}.pkl")
            if hasattr(self.training_env, 'save'):
                self.training_env.save(norm_path)
        return True


def train(args):
    """Main training function"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_v12_{timestamp}"
    log_dir = os.path.join(args.log_dir, run_name)
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"=== PPO Training V12 ===")
    print(f"Log dir: {log_dir}")
    print(f"Timesteps: {args.total_timesteps}")
    print(f"N envs: {args.n_envs}")
    
    # Create vectorized environment
    if args.n_envs > 1:
        env = SubprocVecEnv([make_env(i, args.seed) for i in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_env(0, args.seed)])
    
    # Wrap with VecNormalize (CRITICAL FOR PPO)
    env = VecNormalize(
        env, 
        norm_obs=True, 
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    
    # Create eval environment
    eval_env = DummyVecEnv([make_env(100, args.seed, use_curriculum=False)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # PPO Hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,           # Steps per update per env
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # Slight entropy to encourage exploration
        verbose=1,
        tensorboard_log=log_dir,
        seed=args.seed,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256])
        ),
    )
    
    print(f"\nModel architecture: {model.policy}")
    
    # Callbacks
    callbacks = [
        CurriculumCallback(args.total_timesteps),
        SuccessRateCallback(check_freq=10000),
        CheckpointCallback(
            save_freq=max(50000 // args.n_envs, 1000),
            save_path=checkpoint_dir,
            name_prefix="ppo_v12",
        ),
        SaveNormCallback(checkpoint_dir, save_freq=50000),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(checkpoint_dir, "best"),
            log_path=log_dir,
            eval_freq=max(20000 // args.n_envs, 1000),
            n_eval_episodes=10,
            deterministic=True,
        ),
    ]
    
    # Train
    print("\n=== Starting Training ===")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=True,
    )
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model")
    model.save(final_path)
    env.save(os.path.join(checkpoint_dir, "final_vecnormalize.pkl"))
    
    print(f"\n=== Training Complete ===")
    print(f"Final model saved to: {final_path}")
    
    env.close()
    eval_env.close()


def evaluate(args):
    """Evaluate a trained PPO model"""
    from dual.envs.dual_so101_env import DualSO101PickCubeEnv
    
    print(f"\nLoading PPO model from {args.eval}...")
    
    # Load model
    try:
        model = PPO.load(args.eval)
    except AttributeError:
        print("ERROR: Could not load model. Are you sure this is a PPO model?")
        print("Note: You cannot load a SAC model with train_ppo.py.")
        return

    # Create environment
    env = DualSO101PickCubeEnv(
        render_mode="human" if args.render else None,
        max_episode_steps=500,
        use_curriculum=False,
    )
    
    # Load VecNormalize if available
    # Look for .pkl file
    norm_path = args.eval.replace(".zip", "").replace("final_model", "final_vecnormalize.pkl")
    if not os.path.exists(norm_path):
        norm_path = os.path.join(os.path.dirname(args.eval), "final_vecnormalize.pkl")
    
    vec_env = DummyVecEnv([lambda: env])
    
    if os.path.exists(norm_path):
        print(f"Loading VecNormalize from {norm_path}")
        vec_env = VecNormalize.load(norm_path, vec_env)
        # IMPORTANT for PPO: Turn off training and reward normalization during eval
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print("WARNING: VecNormalize stats not found! Performance may be poor.")
    
    # Run episodes
    n_episodes = args.n_eval_episodes
    successes = 0
    grasps = 0
    lifts = 0
    rewards = []
    
    print(f"\nStarting evaluation over {n_episodes} episodes...")
    
    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward[0]
            steps += 1
            
            if args.render:
                env.render()
        
        info = info[0]
        rewards.append(total_reward)
        
        status = "✓ SUCCESS" if info.get('is_success') else "↑ Lifted" if info.get('was_lifted') else "✗ Failed"
        print(f"Episode {ep}: Reward={total_reward:.1f}, Steps={steps}, {status}")
        
        if info.get('is_success'):
            successes += 1
        if info.get('was_grasped'):
            grasps += 1
        if info.get('was_lifted'):
            lifts += 1
    
    print(f"\n{'='*50}")
    print(f"PPO Results over {n_episodes} episodes:")
    print(f"  Grasp rate:   {grasps}/{n_episodes} ({100*grasps/n_episodes:.1f}%)")
    print(f"  Lift rate:    {lifts}/{n_episodes} ({100*lifts/n_episodes:.1f}%)")
    print(f"  Success rate: {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)")
    print(f"  Avg reward: {np.mean(rewards):.1f}")
    
    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training for Dual-Arm V12")
    
    # Training Arguments
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="dual/logs")
    parser.add_argument("--checkpoint_dir", type=str, default="dual/checkpoints/sb")
    
    # Evaluation Arguments
    parser.add_argument("--eval", type=str, default=None, help="Path to model for evaluation")
    parser.add_argument("--render", action="store_true", help="Render during evaluation")
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    
    args = parser.parse_args()
    
    if args.eval:
        evaluate(args)
    else:
        train(args)