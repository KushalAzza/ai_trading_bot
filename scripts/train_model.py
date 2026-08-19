import os
import json
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from scripts.trading_env import TradingEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np
from stable_baselines3.common.monitor import Monitor
from scripts.hyperparameter_tuning import run_optimization
from scripts.utils import (
    setup_logging,
    ensure_directory_exists,
    load_and_preprocess_data,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    LOGS_DIR,
    OPTUNA_DIR,
    TRANSACTION_COST,
    BASE_LOT_SIZE,
    MAX_EXPOSURE,
    DAILY_RISK_LIMIT,
    INITIAL_BALANCE
)
import logging
from typing import Dict, Any, Optional, Tuple
import optuna
import time

# Set up logging
logger = setup_logging('train_model')

def load_best_hyperparameters():
    """Load the best hyperparameters from the latest Optuna study"""
    try:
        if not os.path.exists(OPTUNA_DIR):
            return None

        # Find the latest study file
        study_files = [f for f in os.listdir(OPTUNA_DIR) if f.endswith('.db')]
        if not study_files:
            return None

        latest_study = max(study_files, key=lambda x: os.path.getctime(os.path.join(OPTUNA_DIR, x)))
        study_path = os.path.join(OPTUNA_DIR, latest_study)

        # Load the study
        study = optuna.load_study(
            study_name=latest_study.replace('.db', ''),
            storage=f"sqlite:///{study_path}"
        )

        # Get best parameters
        best_params = study.best_trial.params
        logger.info(f"Loaded best hyperparameters from study: {latest_study}")
        return best_params

    except Exception as e:
        logger.warning(f"Could not load best hyperparameters: {str(e)}")
        return None

# Default hyperparameters (used if Optuna study is not available)
DEFAULT_PARAMS = {
    'learning_rate': 0.0001,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'total_timesteps': 100000,
    'eval_freq': 10000
}

# Load best hyperparameters or use defaults
HYPERPARAMS = load_best_hyperparameters() or DEFAULT_PARAMS

# Environment hyperparameters
ENV_PARAMS = {
    'initial_balance': INITIAL_BALANCE,
    'transaction_cost': TRANSACTION_COST,
    'max_position': 1,
    'stop_loss_pct': 0.02,
    'max_drawdown_pct': 0.1,
    'base_lot_size': BASE_LOT_SIZE,
    'max_exposure': MAX_EXPOSURE,
    'daily_risk_limit': DAILY_RISK_LIMIT
}

# Create directories if they don't exist
for directory in [PROCESSED_DATA_DIR, MODEL_DIR, LOGS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def split_data(df: pd.DataFrame):
    """Split data into train and validation sets."""
    train_size = int(0.8 * len(df['timestamp'].unique()))
    train_timestamps = df['timestamp'].unique()[:train_size]
    val_timestamps = df['timestamp'].unique()[train_size:]
    train_df = df[df['timestamp'].isin(train_timestamps)]
    val_df = df[df['timestamp'].isin(val_timestamps)]
    return train_df, val_df

def create_env(data_path: str, is_eval: bool = False) -> VecNormalize:
    """Create and configure the trading environment."""
    try:
        # First create the base environment
        base_env = TradingEnv(data=data_path, env_params=ENV_PARAMS)
        
        # Wrap with Monitor
        monitored_env = Monitor(
            base_env,
            filename=f"{LOGS_DIR}/{'eval' if is_eval else 'train'}",
            allow_early_resets=True
        )
        
        # Wrap with DummyVecEnv
        vec_env = DummyVecEnv([lambda: monitored_env])
        
        # Finally wrap with VecNormalize
        env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
            epsilon=1e-08,
            training=not is_eval
        )
        
        return env
        
    except Exception as e:
        logger.error(f"Error creating environment: {str(e)}")
        raise

def load_existing_model(model_path: str, vec_normalize_path: str, train_env) -> tuple:
    """Load existing model and environment if available."""
    try:
        # Check for both .zip extension and without extension
        if not os.path.exists(model_path) and not model_path.endswith('.zip'):
            model_path_with_ext = f"{model_path}.zip"
            if os.path.exists(model_path_with_ext):
                model_path = model_path_with_ext
                logger.info(f"Found model at {model_path}")
        
        if not os.path.exists(model_path):
            logger.warning(f"No existing model found at {model_path}")
            return None, train_env
            
        logger.info(f"Loading model from {model_path}")
        model = PPO.load(
            model_path,
            env=train_env,
            custom_objects={
                "learning_rate": HYPERPARAMS['learning_rate'],
                "n_steps": HYPERPARAMS['n_steps'],
                "batch_size": HYPERPARAMS['batch_size'],
                "gamma": HYPERPARAMS['gamma']
            }
        )
        
        if os.path.exists(vec_normalize_path):
            logger.info(f"Loading VecNormalize stats from {vec_normalize_path}")
            train_env = VecNormalize.load(vec_normalize_path, train_env)
            train_env.training = True  # Enable training mode
            logger.info("Successfully loaded normalization statistics")
        else:
            logger.warning(f"No normalization stats found at {vec_normalize_path}")
        
        # Log model information
        logger.info(f"Successfully loaded model with {model.num_timesteps} previous timesteps")
        return model, train_env
    except Exception as e:
        logger.warning(f"Error loading existing model: {str(e)}")
        logger.warning("Creating new model instead")
        return None, train_env

class MetricLogger(BaseCallback):
    """Custom callback for logging training metrics"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.training_start = None
        self.timestamps = []
        self.rewards = []
        self.value_losses = []
        self.policy_losses = []
        self.explained_variances = []
        
    def _on_training_start(self):
        self.training_start = time.time()
        
    def _on_step(self) -> bool:
        # Log metrics every 1000 steps
        if self.n_calls % 1000 == 0:
            elapsed_time = int(time.time() - self.training_start)
            self.timestamps.append(elapsed_time)
            self.rewards.append(float(self.locals.get('rewards', [0]).mean()))
            self.value_losses.append(float(self.locals.get('value_loss', 0)))
            self.policy_losses.append(float(self.locals.get('policy_loss', 0)))
            self.explained_variances.append(float(self.locals.get('explained_variance', 0)))
            
            logger.info(f"Steps: {self.n_calls}, "
                       f"Mean Reward: {self.rewards[-1]:.2f}, "
                       f"Time: {elapsed_time}s")
        return True

def train_model(
    total_timesteps: int = HYPERPARAMS.get('total_timesteps', 100000),
    learning_rate: float = HYPERPARAMS.get('learning_rate', 0.0001),
    batch_size: int = HYPERPARAMS.get('batch_size', 64),
    n_steps: int = HYPERPARAMS.get('n_steps', 2048),
    n_epochs: int = HYPERPARAMS.get('n_epochs', 10),
    gamma: float = HYPERPARAMS.get('gamma', 0.99),
    eval_freq: int = HYPERPARAMS.get('eval_freq', 10000),
    is_retraining: bool = False
) -> None:
    try:
        logger.info("Starting model training...")
        
        # Create environments
        train_env = create_env(f"{PROCESSED_DATA_DIR}/training_data.csv")
        eval_env = create_env(f"{PROCESSED_DATA_DIR}/training_data.csv", is_eval=True)
        
        # Paths for model files
        model_path = f"{MODEL_DIR}/final_model.zip"
        vec_normalize_path = f"{MODEL_DIR}/vec_normalize.pkl"
        
        # Load existing model if retraining
        if is_retraining and os.path.exists(model_path):
            logger.info("Retraining: Loading existing model...")
            try:
                model = PPO.load(model_path, env=train_env)
                
                if os.path.exists(vec_normalize_path):
                    logger.info("Loading normalization statistics...")
                    train_env = VecNormalize.load(vec_normalize_path, train_env)
                    train_env.training = True
                
                logger.info("Successfully loaded existing model for retraining")
            except Exception as e:
                logger.warning(f"Failed to load existing model: {str(e)}")
                model = None
        else:
            logger.info("Training new model from scratch...")
            model = None
        
        # Create new model if needed
        if model is None:
            model = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                verbose=1
            )
        
        # Set up evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            log_path=f"{LOGS_DIR}/eval",
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        # Create metric logger
        metric_logger = MetricLogger()
        
        # Train the model
        logger.info(f"Training model for {total_timesteps} timesteps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, metric_logger],
            progress_bar=True,
            reset_num_timesteps=not is_retraining
        )
        
        # Save the model and environment
        model.save(model_path)
        train_env.save(vec_normalize_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save training metadata and metrics
        training_metadata = {
            'total_timesteps_trained': model.num_timesteps,
            'last_training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_retraining': is_retraining,
            'hyperparameters': {
                'learning_rate': learning_rate,
                'n_steps': n_steps,
                'batch_size': batch_size,
                'n_epochs': n_epochs,
                'gamma': gamma
            }
        }
        
        metadata_path = f"{MODEL_DIR}/training_metadata.json"
        metrics = {
            'timestamps': metric_logger.timestamps,
            'rewards': metric_logger.rewards,
            'value_losses': metric_logger.value_losses,
            'policy_losses': metric_logger.policy_losses,
            'explained_variances': metric_logger.explained_variances,
            'metadata': training_metadata
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Training metadata and metrics saved to {metadata_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    finally:
        # Clean up environments
        train_env.close()
        eval_env.close()

def quick_validate(model_path, data_path):
    """Quick validation of a trained model."""
    logger.info(f"Starting quick model validation...")
    
    # Load and preprocess data
    data = load_and_preprocess_data(data_path)
    logger.info(f"Loaded and preprocessed data from {data_path} with {len(data)} rows")
    
    # Define environment parameters using constants from utils.py
    env_params = {
        'initial_balance': INITIAL_BALANCE,
        'transaction_cost': TRANSACTION_COST,
        'max_position': 1,
        'stop_loss_pct': 0.02,
        'max_drawdown_pct': 0.1,
        'base_lot_size': BASE_LOT_SIZE,
        'max_exposure': MAX_EXPOSURE,
        'daily_risk_limit': DAILY_RISK_LIMIT
    }
    
    # Create base environment
    base_env = TradingEnv(data, env_params)
    
    # Wrap with Monitor
    env = Monitor(
        base_env,
        filename=f"{LOGS_DIR}/quick_validate",
        allow_early_resets=True
    )
    
    # Load model
    model = PPO.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    
    # Run validation episodes
    episode_rewards = []
    episode_pnls = []
    episode_trades = []
    episode_drawdowns = []
    
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Handle new Gym API format
    
    done = False
    episode_reward = 0
    
    while not done:
        action, _ = model.predict(obs)
        step_result = env.step(action)
        
        # Handle both old and new Gym API formats
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
        
        episode_reward += reward
        
        if done:
            episode_rewards.append(float(episode_reward))
            episode_pnls.append(float(info['balance']))
            episode_trades.append(float(info['trades']))
            episode_drawdowns.append(float(info['drawdown']))
            
            logger.info(f"Episode finished with reward: {episode_reward:.2f}, PnL: {info['balance']:.2f}, "
                       f"Trades: {info['trades']}, Max Drawdown: {info['drawdown']:.2%}")
    
    # Calculate average metrics
    avg_reward = np.mean(episode_rewards)
    avg_pnl = np.mean(episode_pnls)
    avg_trades = np.mean(episode_trades)
    max_drawdown = max(episode_drawdowns)
    
    logger.info(f"\nValidation Results:")
    logger.info(f"Average Reward: {avg_reward:.2f}")
    logger.info(f"Average PnL: {avg_pnl:.2f}")
    logger.info(f"Average Trades per Episode: {avg_trades:.2f}")
    logger.info(f"Maximum Drawdown: {max_drawdown:.2%}")
    
    # Add warnings for poor performance
    if avg_pnl < 0:
        logger.warning(f"Warning: Negative average PnL ({avg_pnl:.2f})")
    if max_drawdown > 0.2:
        logger.warning(f"Warning: High maximum drawdown ({max_drawdown:.2%})")
    if avg_trades < 1:
        logger.warning("Warning: Very low trading activity")

def optimize_hyperparameters(force_optimization=False):
    """Run hyperparameter optimization if no previous study exists or if forced"""
    try:
        if not os.path.exists(OPTUNA_DIR):
            os.makedirs(OPTUNA_DIR)
            logger.info("Created Optuna directory")
        
        study_files = [f for f in os.listdir(OPTUNA_DIR) if f.endswith('.db')]
        
        if force_optimization or not study_files:
            logger.info("Running hyperparameter optimization...")
            run_optimization(n_trials=5, n_jobs=1)
            logger.info("Hyperparameter optimization completed")
        else:
            logger.info("Using existing hyperparameter optimization results")
    except Exception as e:
        logger.error(f"Error during hyperparameter optimization: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train the trading model')
    parser.add_argument('--optimize', action='store_true', help='Force hyperparameter optimization')
    parser.add_argument('--retrain', action='store_true', help='Retrain existing model')
    args = parser.parse_args()
    
    # Run hyperparameter optimization if needed or requested
    optimize_hyperparameters(force_optimization=args.optimize)
    
    # Train the model
    train_model(is_retraining=args.retrain)
