import os
import sys
from pathlib import Path
import optuna
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

# Add the project root directory to Python path for imports
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.trading_env import TradingEnv
from scripts.utils import (
    setup_logging, 
    ensure_directory_exists, 
    PROCESSED_DATA_DIR, 
    MODEL_DIR, 
    LOGS_DIR, 
    OPTUNA_DIR,
    load_and_preprocess_data,
    BASE_LOT_SIZE,
    TRANSACTION_COST,
    INITIAL_BALANCE,
    MAX_EXPOSURE,
    DAILY_RISK_LIMIT
)

# Custom exceptions
class OptimizationError(Exception):
    """Custom exception for optimization errors."""
    pass

# Define optimization parameters
OPTIMIZATION_PARAMS = {
    'model_params': [
        ('n_steps', 'int', (1024, 8192)),
        ('gamma', 'float', (0.9, 0.9999)),
        ('learning_rate', 'float', (1e-6, 1e-3)),
        ('ent_coef', 'float', (1e-8, 0.1)),
        ('clip_range', 'float', (0.1, 0.4)),
        ('n_epochs', 'int', (5, 30)),
        ('gae_lambda', 'float', (0.9, 0.99)),
        ('max_grad_norm', 'float', (0.1, 1.0)),
        ('vf_coef', 'float', (0.1, 0.9)),
        ('total_timesteps', 'int', (50000, 100000))
    ],
    'env_params': [
        ('max_position', 'int', (1, 3)),
        ('stop_loss_pct', 'float', (0.01, 0.05)),
        ('max_drawdown_pct', 'float', (0.05, 0.2))
    ]
}

# Fixed environment parameters
FIXED_ENV_PARAMS = {
    'transaction_cost': TRANSACTION_COST,
    'initial_balance': INITIAL_BALANCE,
    'base_lot_size': BASE_LOT_SIZE,
    'max_exposure': MAX_EXPOSURE,
    'daily_risk_limit': DAILY_RISK_LIMIT,
    'max_trades_per_day': 5  
}

# Set up logging
logger = setup_logging('hyperparameter_tuning')

# Create directories
for directory in [PROCESSED_DATA_DIR, MODEL_DIR, LOGS_DIR, OPTUNA_DIR]:
    ensure_directory_exists(directory)

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate Sharpe ratio from returns.
    
    Args:
        returns: Array of episode returns (total P&L per episode)
        risk_free_rate: Risk-free rate (default: 0.0)
        periods_per_year: Number of trading periods in a year (default: 252 for trading days)
    
    Returns:
        float: Sharpe ratio or -inf if calculation fails
    """
    if len(returns) < 2:
        logger.warning("Not enough returns to calculate Sharpe ratio")
        return float('-inf')
    
    try:
        # Convert rewards to returns (percentage of initial capital)
        returns = np.array(returns, dtype=float)
        
        # Check for invalid values
        if np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
            logger.warning("Invalid values in returns array")
            return float('-inf')
        
        # Convert absolute rewards to percentage returns
        # Assuming initial_balance is 100000 from FIXED_ENV_PARAMS
        percentage_returns = returns / 100000.0
        
        # Calculate excess returns over risk-free rate
        excess_returns = percentage_returns - (risk_free_rate / periods_per_year)
        
        # Calculate mean and standard deviation
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns, ddof=1)  # Use sample standard deviation
        
        if std_return < 1e-8:  # Avoid division by zero with small threshold
            logger.warning("Standard deviation of returns is too small")
            return float('-inf')
        
        # Calculate Sharpe ratio (annualized)
        sharpe = mean_return / std_return * np.sqrt(periods_per_year)
        
        # Check for valid Sharpe ratio
        if np.isnan(sharpe) or np.isinf(sharpe):
            logger.warning(f"Invalid Sharpe ratio calculated: {sharpe}")
            return float('-inf')
        
        logger.info(f"Calculated Sharpe ratio: {sharpe:.4f} (mean: {mean_return:.6f}, std: {std_return:.6f})")
        return sharpe
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio: {str(e)}")
        return float('-inf')

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown from returns."""
    try:
        if len(returns) < 2:
            return 0.0
            
        returns = np.array(returns, dtype=float)
        
        # Check for invalid values
        if np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
            return 1.0  # Maximum drawdown
            
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        
        # Avoid division by zero
        valid_indices = running_max > 0
        if not np.any(valid_indices):
            return 1.0
            
        drawdown = np.zeros_like(running_max)
        drawdown[valid_indices] = (cumulative[valid_indices] - running_max[valid_indices]) / running_max[valid_indices]
        
        return float(np.min(drawdown))
    except Exception as e:
        logger.error(f"Error calculating max drawdown: {str(e)}")
        return 1.0  # Return maximum drawdown on error

def create_temp_datasets(df, train_size=0.8):
    """Create temporary dataset files with proper cleanup.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        train_size (float): Proportion of data for training
        
    Returns:
        tuple: (train_path, val_path)
        
    Raises:
        OptimizationError: If there's an error creating the datasets
    """
    train_path = f'{PROCESSED_DATA_DIR}/temp_train.csv'
    val_path = f'{PROCESSED_DATA_DIR}/temp_val.csv'
    
    try:
        # Split and save data
        split_idx = int(len(df) * train_size)
        train_data = df.iloc[:split_idx]
        val_data = df.iloc[split_idx:]
        
        train_data.to_csv(train_path, index=False)
        val_data.to_csv(val_path, index=False)
        
        return train_path, val_path
        
    except Exception as e:
        # Clean up if error occurs
        for path in [train_path, val_path]:
            if os.path.exists(path):
                os.remove(path)
        raise OptimizationError(f"Failed to create temporary datasets: {str(e)}")

def make_env(base_env):
    """Helper function to create environment"""
    return base_env

def create_environments(train_path, val_path, env_params, gamma):
    """Create training and validation environments.
    
    Args:
        train_path (str): Path to training data
        val_path (str): Path to validation data
        env_params (dict): Environment parameters
        gamma (float): Discount factor
        
    Returns:
        tuple: (train_env, val_env)
        
    Raises:
        OptimizationError: If there's an error creating the environments
    """
    try:
        # Create training environment
        base_train_env = TradingEnv(data=train_path, env_params=env_params)
        monitored_train_env = Monitor(
            base_train_env,
            filename=None,
            allow_early_resets=True
        )
        vec_train_env = DummyVecEnv([lambda: monitored_train_env])
        train_env = VecNormalize(
            vec_train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=gamma,
            epsilon=1e-08
        )

        # Create validation environment
        base_val_env = TradingEnv(data=val_path, env_params=env_params)
        monitored_val_env = Monitor(
            base_val_env,
            filename=None,
            allow_early_resets=True
        )
        vec_val_env = DummyVecEnv([lambda: monitored_val_env])
        val_env = VecNormalize(
            vec_val_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=gamma,
            epsilon=1e-08,
            training=False
        )

        return train_env, val_env

    except Exception as e:
        raise OptimizationError(f"Failed to create environments: {str(e)}")

def optimize_ppo(trial):
    """Optimize PPO hyperparameters using Optuna"""
    temp_files = []
    environments = []
    
    try:
        # Validate data file exists
        data_file = f'{PROCESSED_DATA_DIR}/training_data.csv'
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Training data not found: {data_file}")

        # First suggest n_steps, ensuring it's a multiple of 1024
        n_steps = trial.suggest_int('n_steps', 1024, 8192, step=1024)
        
        # Then suggest batch_size as a fraction of n_steps
        valid_fractions = [1/4, 1/8, 1/16]
        fraction = trial.suggest_categorical('batch_size_fraction', valid_fractions)
        batch_size = int(n_steps * fraction)
        
        # Get model hyperparameters
        model_params = {
            'n_steps': n_steps,
            'batch_size': batch_size,
            'verbose': 1,
            'device': 'cpu'  # Explicitly set device
        }
        
        # Get PPO-specific hyperparameters
        ppo_params = ['gamma', 'learning_rate', 'ent_coef', 'clip_range', 
                     'n_epochs', 'gae_lambda', 'max_grad_norm', 'vf_coef']
        
        for param_name, param_type, param_range in OPTIMIZATION_PARAMS['model_params']:
            if param_name in ppo_params:  # Only include valid PPO parameters
                if param_type == 'categorical':
                    model_params[param_name] = trial.suggest_categorical(param_name, param_range)
                elif param_type == 'int':
                    model_params[param_name] = trial.suggest_int(param_name, *param_range, log=True)
                elif param_type == 'float':
                    model_params[param_name] = trial.suggest_float(param_name, *param_range, log=True)

        # Get total_timesteps separately as it's used in learn(), not init
        total_timesteps = trial.suggest_int('total_timesteps', 50000, 100000, log=True)

        # Log the parameters being used
        logger.info("Model parameters:")
        for key, value in model_params.items():
            logger.info(f"    {key}: {value}")

        # Get environment hyperparameters
        env_params = {}
        for param_name, param_type, param_range in OPTIMIZATION_PARAMS['env_params']:
            if param_type == 'int':
                env_params[param_name] = trial.suggest_int(param_name, *param_range)
            elif param_type == 'float':
                env_params[param_name] = trial.suggest_float(param_name, *param_range)

        # Update env_params with fixed parameters
        env_params.update(FIXED_ENV_PARAMS)
        
        # Log environment parameters
        logger.info("Environment parameters:")
        for key, value in env_params.items():
            logger.info(f"    {key}: {value}")

        try:
            # Load and preprocess data
            df = load_and_preprocess_data(data_file)
            if df.empty:
                logger.error("Loaded data is empty")
                return float('-inf')
            
            # Create temporary datasets
            train_path, val_path = create_temp_datasets(df)
            temp_files.extend([train_path, val_path])

            # Create environments
            train_env, val_env = create_environments(
                train_path, 
                val_path, 
                env_params,
                model_params['gamma']
            )
            environments.extend([train_env, val_env])

            # Set monitor filenames safely
            try:
                train_env.venv.envs[0].filename = f"{LOGS_DIR}/train_trial_{trial.number}"
                val_env.venv.envs[0].filename = f"{LOGS_DIR}/val_trial_{trial.number}"
            except (AttributeError, IndexError) as e:
                logger.warning(f"Could not set monitor filenames: {str(e)}")
                logger.warning("This is not critical and training will continue")

            # Save normalization stats for validation environment
            if not os.path.exists(f"{MODEL_DIR}/vec_normalize.pkl"):
                train_env.save(f"{MODEL_DIR}/vec_normalize.pkl")

            # Create and train model
            model = PPO("MlpPolicy", train_env, **model_params)
            
            # Set up evaluation callback with early stopping
            eval_callback = EvalCallback(
                val_env,
                best_model_save_path=None,
                log_path=None,
                eval_freq=10000,
                deterministic=True,
                render=False,
                n_eval_episodes=5,
                callback_after_eval=StopTrainingOnNoModelImprovement(
                    max_no_improvement_evals=3,
                    min_evals=5,
                    verbose=1
                )
            )

            # Train the model
            try:
                model.learn(
                    total_timesteps=total_timesteps,
                    callback=eval_callback,
                    progress_bar=True
                )
            except Exception as e:
                logger.error(f"Error during training: {str(e)}")
                return float('-inf')

            # Evaluate model performance
            evaluation_results = evaluate_model(model, val_env)
            
            # Return the objective metric (Sharpe ratio)
            sharpe_ratio = evaluation_results['sharpe_ratio']
            
            # Log results
            logger.info(f"Trial {trial.number} completed with Sharpe ratio: {sharpe_ratio}")
            logger.info(f"Mean reward: {evaluation_results['mean_reward']}")
            logger.info(f"Total trades: {evaluation_results['total_trades']}")
            
            if sharpe_ratio == float('-inf') or np.isnan(sharpe_ratio):
                logger.warning("Invalid Sharpe ratio obtained")
                return float('-inf')
                
            return sharpe_ratio

        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {str(e)}")
            logger.error(f"Model parameters: {model_params}")
            logger.error(f"Environment parameters: {env_params}")
            return float('-inf')

    except Exception as e:
        logger.error(f"Unexpected error in trial: {str(e)}")
        return float('-inf')
    finally:
        cleanup(temp_files, environments)

def evaluate_model(model, env, n_episodes=5):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model to evaluate
        env: Environment to evaluate in
        n_episodes: Number of episodes to evaluate
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    all_episode_rewards = []  # Total reward per episode
    all_step_returns = []     # Step-by-step returns for more granular analysis
    all_infos = []
    
    try:
        for episode in range(n_episodes):
            # Handle both old and new gym API
            try:
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    if len(reset_result) == 2:
                        obs, _ = reset_result
                    else:
                        obs = reset_result[0]
                else:
                    obs = reset_result
            except Exception as e:
                logger.error(f"Error during environment reset: {str(e)}")
                return {
                    'mean_reward': float('-inf'),
                    'std_reward': 0,
                    'sharpe_ratio': float('-inf'),
                    'max_drawdown': 1.0,
                    'total_trades': 0
                }
                
            done = False
            episode_reward = 0
            episode_infos = []
            step_returns = []
            initial_balance = 100000.0  # From FIXED_ENV_PARAMS
            
            while not done:
                try:
                    action, _ = model.predict(obs, deterministic=True)
                    step_result = env.step(action)
                    
                    # Handle both old and new gym API
                    if len(step_result) == 5:  # New API: obs, reward, terminated, truncated, info
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:  # Old API: obs, reward, done, info
                        obs, reward, done, info = step_result
                    
                    # Handle reward
                    if isinstance(reward, (list, np.ndarray)):
                        reward = reward[0]
                    
                    # Track step-by-step returns
                    step_returns.append(reward / initial_balance)  # Convert to percentage return
                    
                    # Accumulate episode reward
                    episode_reward += reward
                    
                    # Handle info
                    if isinstance(info, (list, tuple)):
                        info = info[0]
                    episode_infos.append(info)
                    
                except Exception as e:
                    logger.error(f"Error during environment step: {str(e)}")
                    done = True
            
            # Store episode results
            all_episode_rewards.append(episode_reward)
            all_step_returns.extend(step_returns)
            all_infos.extend(episode_infos)
            
            logger.info(f"Episode {episode+1}/{n_episodes} - Reward: {episode_reward:.2f}")

        # Calculate metrics
        episode_rewards_array = np.array(all_episode_rewards)
        
        # Ensure we have valid rewards
        if len(episode_rewards_array) == 0 or np.all(np.isnan(episode_rewards_array)):
            logger.warning("No valid rewards obtained during evaluation")
            return {
                'mean_reward': float('-inf'),
                'std_reward': 0,
                'sharpe_ratio': float('-inf'),
                'max_drawdown': 1.0,
                'total_trades': 0
            }

        # Calculate max_drawdown safely
        max_drawdown = 0.0
        if all_infos:
            drawdowns = [info.get('drawdown', 0.0) for info in all_infos]
            if drawdowns:
                max_drawdown = max(drawdowns)
        
        # Calculate Sharpe ratio using episode returns
        sharpe_ratio = calculate_sharpe_ratio(episode_rewards_array)

        metrics = {
            'mean_reward': float(np.mean(episode_rewards_array)),
            'std_reward': float(np.std(episode_rewards_array)),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'total_trades': sum(1 for info in all_infos if info.get('trades', 0) > 0)
        }
        
        # Log metrics
        logger.info("Evaluation metrics:")
        for key, value in metrics.items():
            if key in ['max_drawdown']:
                logger.info(f"{key}: {value:.2%}")
            else:
                logger.info(f"{key}: {value:.2f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        return {
            'mean_reward': float('-inf'),
            'std_reward': 0,
            'sharpe_ratio': float('-inf'),
            'max_drawdown': 1.0,
            'total_trades': 0
        }

def cleanup(temp_files, environments):
    """Clean up temporary files and environments.
    
    Args:
        temp_files (list): List of temporary file paths
        environments (list): List of environments to close
    """
    # Remove temporary files
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {file_path}: {e}")

    # Close environments
    for env in environments:
        try:
            env.close()
        except Exception as e:
            logger.warning(f"Failed to close environment: {e}")

def create_visualizations(study, viz_dir):
    """Create optimization visualizations with proper error handling.
    
    Args:
        study: Optuna study object
        viz_dir (str): Directory to save visualizations
    """
    try:
        # Check if study has any completed trials with valid values
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        valid_trials = [t for t in completed_trials if not np.isnan(t.value) and not np.isinf(t.value)]
        
        if not valid_trials:
            logger.warning("No valid trials to visualize")
            return
            
        # Basic plots
        try:
            plot_optimization_history(study).write_image(
                f"{viz_dir}/optimization_history.png"
            )
            plot_param_importances(study).write_image(
                f"{viz_dir}/param_importances.png"
            )
        except Exception as e:
            logger.warning(f"Could not create basic plots: {str(e)}")
        
        # Advanced plots with dependency check
        try:
            import plotly
            from optuna.visualization import plot_parallel_coordinate
            plot_parallel_coordinate(study).write_image(
                f"{viz_dir}/parallel_coordinate.png"
            )
        except ImportError as e:
            logger.warning(f"Could not create parallel coordinate plot: {e}")
            logger.warning("Install plotly for additional visualizations")
        except Exception as e:
            logger.warning(f"Error creating parallel coordinate plot: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        logger.warning("Visualization creation failed but optimization results are still valid")

def run_optimization(n_trials=5, n_jobs=1, timeout=None):
    """Run hyperparameter optimization with Optuna"""
    try:
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        # Create study name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"ppo_optimization_{timestamp}"
        storage_name = f"sqlite:///{OPTUNA_DIR}/{study_name}.db"
        
        # Create and configure the study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=False,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        )
        
        # Run optimization
        study.optimize(
            optimize_ppo,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Get best trial
        best_trial = study.best_trial
        logger.info(f"Best trial: {best_trial.number}")
        logger.info(f"Best value (Sharpe ratio): {best_trial.value}")
        logger.info("Best hyperparameters:")
        
        # Save best parameters
        best_params = best_trial.params
        for key, value in best_params.items():
            logger.info(f"    {key}: {value}")
        
        # Add fixed parameters
        best_params.update(FIXED_ENV_PARAMS)
        
        # Save best parameters to file
        params_path = f"{OPTUNA_DIR}/{study_name}_best_params.json"
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"Best parameters saved to {params_path}")
        
        # Save study for later analysis
        joblib.dump(study, f"{OPTUNA_DIR}/{study_name}_study.pkl")
        
        # Create visualization directory
        viz_dir = f"{OPTUNA_DIR}/{study_name}_visualizations"
        ensure_directory_exists(viz_dir)
        
        # Create visualizations
        create_visualizations(study, viz_dir)
        
        return best_params
        
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        raise

def train_with_best_params(best_params=None, study_path=None):
    """Train a model with the best parameters from optimization"""
    try:
        # Load best parameters from study if not provided
        if best_params is None:
            if study_path is None:
                study_files = [f for f in os.listdir(OPTUNA_DIR) if f.endswith('.db')]
                if not study_files:
                    raise ValueError("No optimization study found")
                study_path = os.path.join(OPTUNA_DIR, study_files[-1])
            
            study = optuna.load_study(
                study_name="trading_optimization",
                storage=f"sqlite:///{study_path}"
            )
            best_params = study.best_params
            logger.info(f"Loaded best parameters from {study_path}")
        
        # Extract environment parameters
        env_params = {k: best_params.get(k, FIXED_ENV_PARAMS.get(k)) for k in [
            'max_position',
            'stop_loss_pct',
            'max_drawdown_pct',
            'transaction_cost',
            'initial_balance',
            'base_lot_size',
            'max_exposure',
            'daily_risk_limit',
            'max_trades_per_day'
        ]}
        
        # Save environment parameters
        env_params_path = f"{MODEL_DIR}/env_params.json"
        with open(env_params_path, 'w') as f:
            json.dump(env_params, f, indent=4)
        
        logger.info(f"Environment parameters saved to {env_params_path}")
        logger.info("You can now run train_model.py to train with these parameters")
        
    except Exception as e:
        logger.error(f"Error training with best parameters: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='Hyperparameter optimization for trading bot')
        parser.add_argument('--trials', type=int, default=5, help='Number of trials')
        parser.add_argument('--jobs', type=int, default=1, help='Number of parallel jobs')
        parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
        parser.add_argument('--train', action='store_true', help='Train with best parameters after optimization')
        args = parser.parse_args()
        
        # Run optimization
        best_params = run_optimization(n_trials=args.trials, n_jobs=args.jobs, timeout=args.timeout)
        
        # Train with best parameters if requested
        if args.train and best_params:
            train_with_best_params(best_params)
            
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        raise 