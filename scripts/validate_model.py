import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from scripts.trading_env import TradingEnv
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from scripts.utils import (
    setup_logging,
    ensure_directory_exists,
    save_trades_to_csv,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    LOGS_DIR,
    OPTUNA_DIR,
    TRANSACTION_COST,
    INITIAL_BALANCE,
    BASE_LOT_SIZE,
    MAX_EXPOSURE,
    DAILY_RISK_LIMIT
)
import json
from stable_baselines3.common.monitor import Monitor

logger = setup_logging('validate_model')

def create_validation_env(data_path: str) -> VecNormalize:
    """Create a validation environment with the same parameters as training."""
    try:
        # Load best parameters if available
        best_params_file = max(
            [f for f in os.listdir(OPTUNA_DIR) if f.endswith('_best_params.json')],
            key=lambda x: os.path.getctime(os.path.join(OPTUNA_DIR, x))
        )
        with open(os.path.join(OPTUNA_DIR, best_params_file), 'r') as f:
            best_params = json.load(f)
            
        # Create environment parameters
        env_params = {
            'initial_balance': best_params.get('initial_balance', 100000.0),
            'transaction_cost': TRANSACTION_COST,
            'max_position': best_params.get('max_position', 1),
            'stop_loss_pct': best_params.get('stop_loss_pct', 0.02),
            'max_drawdown_pct': best_params.get('max_drawdown_pct', 0.1),
            'base_lot_size': BASE_LOT_SIZE,
            'max_exposure': MAX_EXPOSURE,
            'daily_risk_limit': DAILY_RISK_LIMIT
        }
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Could not load best parameters: {str(e)}. Using defaults from train_model.py")
        from scripts.train_model import ENV_PARAMS
        env_params = ENV_PARAMS

    # Create and return environment
    base_env = TradingEnv(data=data_path, env_params=env_params)
    monitored_env = Monitor(base_env, filename=f"{LOGS_DIR}/validation", allow_early_resets=True)
    vec_env = DummyVecEnv([lambda: monitored_env])
    return VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-08,
        training=False
    )

def run_time_series_validation(
    n_splits: int = 5,
    test_size: int = 30,  # Number of days in each test fold
    plot_results: bool = True
) -> None:
    """Run validation using TimeSeriesSplit for proper time series evaluation."""
    try:
        logger.info(f"Starting time series cross-validation with {n_splits} splits")
        
        # Load data
        df = pd.read_csv(f"{PROCESSED_DATA_DIR}/training_data.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by date to get unique trading days
        df['date'] = df['timestamp'].dt.date
        unique_dates = sorted(df['date'].unique())
        
        if len(unique_dates) < n_splits * 2:
            logger.warning(f"Not enough data for {n_splits} splits. Reducing to {len(unique_dates)//2} splits.")
            n_splits = max(2, len(unique_dates) // 2)
        
        # Create TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Prepare results storage
        all_metrics = []
        fold_results = []
        
        # Load model
        model_path = f"{MODEL_DIR}/final_model.zip"
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}")
            return
            
        # Create directory for validation results
        validation_dir = f"{LOGS_DIR}/time_series_validation"
        ensure_directory_exists(validation_dir)
        
        # Run validation for each fold
        for fold, (train_idx, test_idx) in enumerate(tscv.split(unique_dates)):
            logger.info(f"Validating fold {fold+1}/{n_splits}")
            
            # Get train and test dates
            train_dates = [unique_dates[i] for i in train_idx]
            test_dates = [unique_dates[i] for i in test_idx]
            
            # Limit test dates to the specified test_size
            test_dates = test_dates[-test_size:] if len(test_dates) > test_size else test_dates
            
            # Create test dataframe
            test_df = df[df['date'].isin(test_dates)].copy()
            
            # Save test data for this fold
            test_data_path = f"{validation_dir}/fold_{fold+1}_test_data.csv"
            test_df.to_csv(test_data_path, index=False)
            
            # Create validation environment
            env = create_validation_env(test_data_path)
            
            # Load model and normalization stats
            model = PPO.load(model_path)
            vec_normalize_path = f"{MODEL_DIR}/vec_normalize.pkl"
            if os.path.exists(vec_normalize_path):
                env = VecNormalize.load(vec_normalize_path, env)
                env.training = False  # Don't update normalization stats during evaluation
                env.norm_reward = False  # Don't normalize rewards during evaluation
            
            # Run validation episodes
            n_episodes = 1  # For time series, we typically do one full run through the test data
            total_reward = 0
            all_trades = []
            
            for episode in range(n_episodes):
                # Handle both old and new Gym API return values
                reset_result = env.reset()
                if isinstance(reset_result, tuple) and len(reset_result) == 2:
                    obs, _ = reset_result
                else:
                    obs = reset_result
                    
                done = False
                episode_reward = 0
                episode_trades = []
                step = 0
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    
                    # Handle both old and new Gym API return values
                    step_result = env.step(action)
                    if len(step_result) == 5:  # New Gym API
                        next_obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:  # Old Gym API
                        next_obs, reward, done, info = step_result
                    
                    # Record trade if action was taken
                    if action != 0:  # Not hold
                        trade_info = info[0].copy()
                        trade_info.update({
                            'fold': fold + 1,
                            'step': step,
                            'action': 'Buy' if action == 1 else 'Sell',
                            'reward': float(reward),
                            'date': test_dates[min(step, len(test_dates)-1)]
                        })
                        episode_trades.append(trade_info)
                    
                    obs = next_obs
                    episode_reward += reward
                    step += 1
                
                total_reward += episode_reward
                all_trades.extend(episode_trades)
            
            # Calculate metrics for this fold
            avg_reward = total_reward / n_episodes
            
            # Calculate P&L, win rate, and other metrics
            if all_trades:
                pnl = all_trades[-1].get('balance', 0) - all_trades[0].get('balance', 0) if all_trades else 0
                win_trades = sum(1 for t in all_trades if t.get('reward', 0) > 0)
                win_rate = (win_trades / len(all_trades)) * 100 if all_trades else 0
                max_drawdown = min([t.get('drawdown', 0) for t in all_trades]) if all_trades else 0
                sharpe = avg_reward / (abs(max_drawdown) + 1e-6)  # Avoid division by zero
            else:
                pnl = 0
                win_rate = 0
                max_drawdown = 0
                sharpe = 0
            
            # Save trades for this fold
            trades_path = f"{validation_dir}/fold_{fold+1}_trades.csv"
            if all_trades:
                pd.DataFrame(all_trades).to_csv(trades_path, index=False)
            
            # Log results
            logger.info(f"Fold {fold+1} Results:")
            logger.info(f"  Test Period: {min(test_dates)} to {max(test_dates)}")
            logger.info(f"  Average Reward: {avg_reward:.2f}")
            logger.info(f"  P&L: ₹{pnl:.2f}")
            logger.info(f"  Win Rate: {win_rate:.2f}%")
            logger.info(f"  Max Drawdown: {max_drawdown:.2f}")
            logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
            
            # Store results for this fold
            fold_result = {
                'fold': fold + 1,
                'start_date': min(test_dates),
                'end_date': max(test_dates),
                'avg_reward': avg_reward,
                'pnl': pnl,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe,
                'n_trades': len(all_trades)
            }
            fold_results.append(fold_result)
            
            # Clean up
            env.close()
        
        # Aggregate results across all folds
        results_df = pd.DataFrame(fold_results)
        results_df.to_csv(f"{validation_dir}/all_folds_results.csv", index=False)
        
        # Calculate and log overall metrics
        logger.info("\n===== Overall Time Series Validation Results =====")
        logger.info(f"Average P&L: ₹{results_df['pnl'].mean():.2f} (±{results_df['pnl'].std():.2f})")
        logger.info(f"Average Win Rate: {results_df['win_rate'].mean():.2f}% (±{results_df['win_rate'].std():.2f}%)")
        logger.info(f"Average Sharpe Ratio: {results_df['sharpe_ratio'].mean():.2f} (±{results_df['sharpe_ratio'].std():.2f})")
        logger.info(f"Average Max Drawdown: {results_df['max_drawdown'].mean():.2f} (±{results_df['max_drawdown'].std():.2f})")
        
        # Plot results if requested
        if plot_results:
            # Create plots directory
            plots_dir = f"{validation_dir}/plots"
            ensure_directory_exists(plots_dir)
            
            # Plot P&L across folds
            plt.figure(figsize=(12, 6))
            plt.bar(results_df['fold'], results_df['pnl'])
            plt.axhline(y=results_df['pnl'].mean(), color='r', linestyle='-', label=f'Mean: {results_df["pnl"].mean():.2f}')
            plt.xlabel('Fold')
            plt.ylabel('P&L (₹)')
            plt.title('P&L Across Time Series Folds')
            plt.legend()
            plt.savefig(f"{plots_dir}/pnl_by_fold.png")
            
            # Plot win rate across folds
            plt.figure(figsize=(12, 6))
            plt.bar(results_df['fold'], results_df['win_rate'])
            plt.axhline(y=results_df['win_rate'].mean(), color='r', linestyle='-', label=f'Mean: {results_df["win_rate"].mean():.2f}%')
            plt.xlabel('Fold')
            plt.ylabel('Win Rate (%)')
            plt.title('Win Rate Across Time Series Folds')
            plt.legend()
            plt.savefig(f"{plots_dir}/win_rate_by_fold.png")
            
            # Plot Sharpe ratio across folds
            plt.figure(figsize=(12, 6))
            plt.bar(results_df['fold'], results_df['sharpe_ratio'])
            plt.axhline(y=results_df['sharpe_ratio'].mean(), color='r', linestyle='-', label=f'Mean: {results_df["sharpe_ratio"].mean():.2f}')
            plt.xlabel('Fold')
            plt.ylabel('Sharpe Ratio')
            plt.title('Sharpe Ratio Across Time Series Folds')
            plt.legend()
            plt.savefig(f"{plots_dir}/sharpe_by_fold.png")
            
            # Plot number of trades across folds
            plt.figure(figsize=(12, 6))
            plt.bar(results_df['fold'], results_df['n_trades'])
            plt.axhline(y=results_df['n_trades'].mean(), color='r', linestyle='-', label=f'Mean: {results_df["n_trades"].mean():.2f}')
            plt.xlabel('Fold')
            plt.ylabel('Number of Trades')
            plt.title('Number of Trades Across Time Series Folds')
            plt.legend()
            plt.savefig(f"{plots_dir}/trades_by_fold.png")
            
            logger.info(f"Plots saved to {plots_dir}")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Error in time series validation: {str(e)}")
        raise

def run_validation(
    n_episodes: int = 10,
    save_trades: bool = True
) -> Tuple[float, float, float, List[Dict]]:
    """Run validation episodes and return performance metrics."""
    try:
        logger.info("Loading model and environment...")
        
        # Load the model and environment
        env_params = {
            'initial_balance': 100000.0,
            'transaction_cost': 50.0,
            'max_position': 1,
            'stop_loss_pct': 0.02,
            'max_drawdown_pct': 0.1
        }
        env = DummyVecEnv([lambda: TradingEnv(f"{PROCESSED_DATA_DIR}/processed_data.csv", env_params)])
        env = VecNormalize.load(f"{MODEL_DIR}/vec_normalize.pkl", env)
        env.training = False
        env.norm_reward = False
        
        model = PPO.load(f"{MODEL_DIR}/final_model.zip")
        
        # Initialize metrics
        total_pnl = 0.0
        total_trades = 0
        max_drawdown = 0.0
        trades_list = []
        
        logger.info(f"Running {n_episodes} validation episodes...")
        
        for episode in range(n_episodes):
            # Handle both old and new Gym API return values
            reset_result = env.reset()
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs, _ = reset_result
            else:
                obs = reset_result
                
            done = False
            episode_pnl = 0.0
            episode_trades = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                # Handle both old and new Gym API return values
                step_result = env.step(action)
                if len(step_result) == 5:  # New Gym API
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:  # Old Gym API
                    obs, reward, done, info = step_result
                
                # Update metrics
                episode_pnl = info[0].get('pnl', 0)
                max_drawdown = min(max_drawdown, info[0].get('drawdown', 0))
                
                # Collect trade information
                if info[0].get('trade_closed', False):
                    trade_info = info[0].get('last_trade', {})
                    if trade_info:
                        episode_trades.append(trade_info)
                        
                # Add logging to track actions
                logger.info(f"Action taken: {action}")
            
            total_pnl += episode_pnl
            total_trades += len(episode_trades)
            trades_list.extend(episode_trades)
            
            logger.info(f"Episode {episode + 1}: P&L: ₹{episode_pnl:.2f}, "
                       f"Trades: {len(episode_trades)}")
                       
        # Calculate metrics
        avg_pnl = total_pnl / n_episodes
        win_rate = sum(1 for t in trades_list if t['pnl'] > 0) / max(len(trades_list), 1)
        sharpe_ratio = avg_pnl / (abs(max_drawdown) + 1e-6)
        
        # Save trades if requested
        if save_trades and trades_list:
            trades_df = pd.DataFrame(trades_list)
            save_trades_to_csv(trades_list, f"{LOGS_DIR}/validation_trades.csv")
            
        return avg_pnl, win_rate, sharpe_ratio, trades_list
        
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        raise
    finally:
        env.close()

def analyze_results(
    avg_pnl: float,
    win_rate: float,
    sharpe_ratio: float,
    trades: List[Dict]
) -> None:
    """Analyze and log validation results."""
    try:
        logger.info("\nValidation Results:")
        logger.info(f"Average P&L per Episode: ₹{avg_pnl:.2f}")
        logger.info(f"Total Trades: {len(trades)}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        if trades:
            # Calculate trade statistics
            pnl_series = pd.Series([t['pnl'] for t in trades])
            holding_times = pd.Series([
                (pd.to_datetime(t['exit_time']) - pd.to_datetime(t['entry_time'])).total_seconds() / 60
                for t in trades
            ])
            
            logger.info("\nTrade Statistics:")
            logger.info(f"Average Trade P&L: ₹{pnl_series.mean():.2f}")
            logger.info(f"Median Trade P&L: ₹{pnl_series.median():.2f}")
            logger.info(f"Std Dev of P&L: ₹{pnl_series.std():.2f}")
            logger.info(f"Average Hold Time: {holding_times.mean():.1f} minutes")
            logger.info(f"Median Hold Time: {holding_times.median():.1f} minutes")
            
            # Analyze trade distribution
            profit_trades = pnl_series[pnl_series > 0]
            loss_trades = pnl_series[pnl_series < 0]
            
            if len(profit_trades) > 0:
                logger.info("\nProfit Trade Statistics:")
                logger.info(f"Average Profit: ₹{profit_trades.mean():.2f}")
                logger.info(f"Max Profit: ₹{profit_trades.max():.2f}")
                
            if len(loss_trades) > 0:
                logger.info("\nLoss Trade Statistics:")
                logger.info(f"Average Loss: ₹{loss_trades.mean():.2f}")
                logger.info(f"Max Loss: ₹{loss_trades.min():.2f}")
                
        # Check for warning signs
        if avg_pnl <= 0:
            logger.warning("\n⚠️ Warning: Negative average P&L")
        if win_rate < 0.4:
            logger.warning("⚠️ Warning: Low win rate")
        if sharpe_ratio < 1:
            logger.warning("⚠️ Warning: Poor risk-adjusted returns")
            
    except Exception as e:
        logger.error(f"Error analyzing results: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Run both validation methods
        logger.info("Running standard validation...")
        avg_pnl, win_rate, sharpe_ratio, trades = run_validation()
        analyze_results(avg_pnl, win_rate, sharpe_ratio, trades)
        
        logger.info("\nRunning time series cross-validation...")
        results_df = run_time_series_validation()
        
        # Compare results
        logger.info("\n===== Validation Methods Comparison =====")
        logger.info(f"Standard Validation - P&L: ₹{avg_pnl:.2f}, Win Rate: {win_rate:.2f}%, Sharpe: {sharpe_ratio:.2f}")
        logger.info(f"Time Series CV - Avg P&L: ₹{results_df['pnl'].mean():.2f}, Avg Win Rate: {results_df['win_rate'].mean():.2f}%, Avg Sharpe: {results_df['sharpe_ratio'].mean():.2f}")
        
        # Provide recommendation
        ts_pnl = results_df['pnl'].mean()
        if ts_pnl < avg_pnl * 0.8:
            logger.warning("⚠️ Time series validation shows significantly worse performance than standard validation.")
            logger.warning("   This suggests the model may be overfitting to specific market conditions.")
        elif ts_pnl > avg_pnl * 1.2:
            logger.info("✅ Time series validation shows better performance than standard validation.")
            logger.info("   This suggests the model generalizes well to new market conditions.")
        else:
            logger.info("✓ Time series validation results are consistent with standard validation.")
            logger.info("   This suggests the model has stable performance across different time periods.")
            
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        raise 