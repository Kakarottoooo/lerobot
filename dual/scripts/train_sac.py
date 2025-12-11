"""
SAC Training Script for Dual-Arm SO-101 V12

Key changes for V12:
- Environment now penalizes hovering in place
- Stronger directional rewards for transport
- Descent reward when over target
- ADDED: Ability to load pretrained model for fine-tuning
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3.common.logger import configure
from stable_baselines3 import SAC
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

        # Import here to avoid issues
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
        
        # Unwrap to get base environment
        vec_env = self.training_env
        
        # Handle VecNormalize wrapper
        if hasattr(vec_env, 'venv'):
            vec_env = vec_env.venv
            
        # Now we have VecEnv, iterate through envs
        if hasattr(vec_env, 'envs'):
            for env in vec_env.envs:
                # Unwrap Monitor
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
        # Check for episode ends
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if 'episode' in info:
                    # Episode ended
                    self.episode_successes.append(info.get('is_success', False))
                    self.episode_grasps.append(info.get('was_grasped', False))
                    self.episode_lifts.append(info.get('was_lifted', False))
        
        if self.num_timesteps % self.check_freq == 0 and len(self.episode_successes) > 0:
            n = len(self.episode_successes)
            success_rate = sum(self.episode_successes) / n * 100
            grasp_rate = sum(self.episode_grasps) / n * 100
            lift_rate = sum(self.episode_lifts) / n * 100
            
            print(f"\n[{self.num_timesteps}] Episodes: {n}")
            print(f"  Grasp: {grasp_rate:.1f}% | Lift: {lift_rate:.1f}% | Success: {success_rate:.1f}%")
            
            # Log to tensorboard
            self.logger.record("custom/success_rate", success_rate)
            self.logger.record("custom/grasp_rate", grasp_rate)
            self.logger.record("custom/lift_rate", lift_rate)
            
            # Reset
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
                if self.verbose:
                    print(f"Saved VecNormalize to {norm_path}")
        return True


def train(args):
    """Main training function"""
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sac_v12_{timestamp}"
    log_dir = os.path.join(args.log_dir, run_name)
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"=== SAC Training V12 ===")
    print(f"Log dir: {log_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"N envs: {args.n_envs}")
    
    # Create vectorized environment
    if args.n_envs > 1:
        env = SubprocVecEnv([make_env(i, args.seed) for i in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_env(0, args.seed)])
    
    # Wrap with VecNormalize
    # NOTE: If loading a model, ideally we should load the old VecNormalize stats too
    # but starting fresh normalization is often okay for fine-tuning if clipping is consistent.
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
    
    # === FINE-TUNING LOGIC ===
    # === FINE-TUNING LOGIC ===
    if args.load:
        print(f"\n!!! LOADING PRETRAINED MODEL: {args.load} !!!")
        try:
            # Load the model but attach it to the new environment
            model = SAC.load(args.load, env=env)
            
            # === FIX STARTS HERE ===
            # We must configure a NEW logger for the new directory
            # giving it None caused the crash.
            new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
            model.set_logger(new_logger)
            # =======================
            
            # Resume learning immediately
            model.learning_starts = 1000 
            
            print("Model loaded successfully. Resuming training...")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return
    else:
        # ... (keep the existing 'Starting Fresh' else block as it is) ...
        print("\nStarting Fresh Training (No model loaded)")
        # SAC hyperparameters (tuned for manipulation)
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            target_entropy="auto",
            policy_kwargs=dict(
                net_arch=[256, 256, 256],
            ),
            verbose=1,
            tensorboard_log=log_dir,
            seed=args.seed,
        )
    
    print(f"\nModel architecture: {model.policy}")
    
    # Callbacks
    callbacks = [
        CurriculumCallback(args.total_timesteps),
        SuccessRateCallback(check_freq=10000),
        CheckpointCallback(
            save_freq=max(50000 // args.n_envs, 1000),
            save_path=checkpoint_dir,
            name_prefix="sac_v12",
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
    
    # Copy to latest
    latest_dir = os.path.join(args.checkpoint_dir, "latest")
    os.makedirs(latest_dir, exist_ok=True)
    model.save(os.path.join(latest_dir, "final_model"))
    env.save(os.path.join(latest_dir, "final_vecnormalize.pkl"))
    print(f"Also saved to: {latest_dir}")
    
    env.close()
    eval_env.close()


def evaluate(args):
    """Evaluate a trained model"""
    from dual.envs.dual_so101_env import DualSO101PickCubeEnv
    
    print(f"\nLoading model from {args.eval}...")
    
    # Load model
    model = SAC.load(args.eval)
    
    # Create environment
    env = DualSO101PickCubeEnv(
        render_mode="human" if args.render else None,
        max_episode_steps=500,
        use_curriculum=False,
    )
    
    # Load VecNormalize if available
    norm_path = args.eval.replace(".zip", "").replace("final_model", "final_vecnormalize.pkl")
    if not os.path.exists(norm_path):
        norm_path = os.path.join(os.path.dirname(args.eval), "final_vecnormalize.pkl")
    
    vec_env = DummyVecEnv([lambda: env])
    if os.path.exists(norm_path):
        print(f"Loading VecNormalize from {norm_path}")
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    
    # Run episodes
    n_episodes = args.n_eval_episodes
    successes = 0
    grasps = 0
    lifts = 0
    rewards = []
    
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
    print(f"Results over {n_episodes} episodes:")
    print(f"  Grasp rate:   {grasps}/{n_episodes} ({100*grasps/n_episodes:.1f}%)")
    print(f"  Lift rate:    {lifts}/{n_episodes} ({100*lifts/n_episodes:.1f}%)")
    print(f"  Success rate: {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)")
    print(f"  Avg reward: {np.mean(rewards):.1f}")
    print(f"  Std reward: {np.std(rewards):.1f}")
    print(f"  Min/Max: {min(rewards):.1f} / {max(rewards):.1f}")
    
    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAC Training for Dual-Arm V12")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="dual/logs")
    parser.add_argument("--checkpoint_dir", type=str, default="dual/checkpoints/sb")
    parser.add_argument("--eval", type=str, default=None, help="Path to model for evaluation")
    parser.add_argument("--render", action="store_true", help="Render during evaluation")
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--load", type=str, default=None, help="Path to pretrained model to fine-tune")
    
    args = parser.parse_args()
    
    if args.eval:
        evaluate(args)
    else:
        train(args)