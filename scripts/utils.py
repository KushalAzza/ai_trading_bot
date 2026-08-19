"""Utility functions for the trading bot."""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.trading_env import TradingEnv

from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Directory Constants
PROCESSED_DATA_DIR = 'processed_data'
UNPROCESSED_DATA_DIR = 'pre_processed_data'
MODEL_DIR = 'models'
LOGS_DIR = 'logs'
OPTUNA_DIR = 'optuna'
DEPLOYMENT_DIR = os.path.join(LOGS_DIR, 'deployment')
TRADE_HISTORY_DIR = os.path.join(DEPLOYMENT_DIR, 'trade_history')
PERFORMANCE_DIR = os.path.join(DEPLOYMENT_DIR, 'performance')

# Default precision for float values
DEFAULT_FLOAT_PRECISION = 6

def setup_logging(script_name: str) -> logging.Logger:
    """Set up logging with both file and console handlers."""
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
        
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(f'{LOGS_DIR}/{script_name}.log')
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(console_handler)
    
    return logger

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess data with proper error handling."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert option_type to numeric (CE=1, PE=0)
        df['option_type'] = (df['option_type'] == 'CE').astype(int)
        
        # Ensure numeric columns
        numeric_columns = [
            'nifty_ltp', 'nifty_high', 'nifty_low', 'india_vix',
            'strike_price', 'bid_price', 'ask_price', 'volume',
            'oi', 'iv', 'delta', 'theta', 'gamma', 'vega',
            'macd', 'macd_signal', 'macd_histogram', 
            'option_rsi', 'nifty_rsi', 'atr', 'sma'
        ]
                         
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Fill missing values using forward fill then backward fill
        df = df.ffill().bfill().fillna(0)
        
        return df
        
    except Exception as e:
        logger = setup_logging('data_preprocessing')
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def ensure_directory_exists(directory: str) -> None:
    """Ensure a directory exists, create if it doesn't."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def save_trades_to_csv(trades: List[Dict], filename: str) -> None:
    """Save trades to CSV with proper error handling."""
    try:
        ensure_directory_exists(os.path.dirname(filename))
        df = pd.DataFrame(trades)
        
        # Append if file exists, create new if it doesn't
        mode = 'a' if os.path.exists(filename) else 'w'
        header = not os.path.exists(filename)
        
        df.to_csv(filename, mode=mode, header=header, index=False)
    except Exception as e:
        logger = setup_logging('trade_logging')
        logger.error(f"Error saving trades to CSV: {str(e)}")
        raise

def split_data(df: pd.DataFrame, train_size: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and validation sets based on timestamps.
    
    Args:
        df: Input DataFrame with a 'timestamp' column
        train_size: Proportion of data for training (default: 0.8)
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and validation DataFrames
    """
    train_size = int(train_size * len(df['timestamp'].unique()))
    train_timestamps = df['timestamp'].unique()[:train_size]
    val_timestamps = df['timestamp'].unique()[train_size:]
    train_df = df[df['timestamp'].isin(train_timestamps)]
    val_df = df[df['timestamp'].isin(val_timestamps)]
    return train_df, val_df

def safe_get(data: Dict, path: List[str], default: Any = None) -> Any:
    """Safely navigate nested dictionaries.
    
    Args:
        data: Dictionary to navigate
        path: List of keys to follow
        default: Default value if path not found
        
    Returns:
        Any: Value at the path or default if not found
    """
    for key in path:
        if not isinstance(data, dict):
            return default
        data = data.get(key, default)
        if data is None:
            return default
    return data

def safe_float(value: Any, default: float = 0.0, precision: int = None) -> float:
    """Safely convert value to float with optional precision.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        precision: Number of decimal places (optional)
        
    Returns:
        float: Converted value or default
    """
    try:
        result = float(value) if value is not None else default
        return round(result, precision if precision is not None else DEFAULT_FLOAT_PRECISION)
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        int: Converted value or default
    """
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def calculate_sharpe_ratio(returns: np.ndarray, 
                         risk_free_rate: float = 0.0, 
                         periods_per_year: int = 252) -> float:
    """Calculate annualized Sharpe ratio from returns.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (default: 0.0)
        periods_per_year: Number of periods in a year (default: 252 for trading days)
    
    Returns:
        float: Calculated Sharpe ratio or -inf if calculation is invalid
    """
    try:
        if len(returns) < 2:
            return float('-inf')
            
        # Convert to numpy array if not already
        returns = np.array(returns)
        
        # Check for invalid values
        if np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
            return float('-inf')
            
        mean_return = calculate_mean(returns)
        std_return = calculate_std(returns)
        
        # Avoid division by zero
        if std_return < 1e-8:
            return float('-inf')
            
        # Calculate annualized Sharpe ratio
        sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(periods_per_year)
        
        return float(sharpe)
        
    except Exception as e:
        logger = setup_logging('calculations')
        logger.error(f"Error calculating Sharpe ratio: {str(e)}")
        return float('-inf')

def calculate_mean(values: np.ndarray) -> float:
    """Calculate mean of values with proper error handling.
    
    Args:
        values: Array of numeric values
        
    Returns:
        float: Calculated mean or 0.0 if calculation fails
    """
    try:
        return float(np.mean(values))
    except Exception as e:
        logger = setup_logging('calculations')
        logger.error(f"Error calculating mean: {str(e)}")
        return 0.0

def calculate_std(values: np.ndarray, ddof: int = 1) -> float:
    """Calculate standard deviation with proper error handling.
    
    Args:
        values: Array of numeric values
        ddof: Delta degrees of freedom (default: 1 for sample standard deviation)
        
    Returns:
        float: Calculated standard deviation or 0.0 if calculation fails
    """
    try:
        if len(values) <= ddof:
            return 0.0
        return float(np.std(values, ddof=ddof))
    except Exception as e:
        logger = setup_logging('calculations')
        logger.error(f"Error calculating standard deviation: {str(e)}")
        return 0.0

def calculate_reward_metrics(rewards: np.ndarray) -> Dict[str, float]:
    """Calculate reward-related metrics.
    
    Args:
        rewards: Array of rewards
        
    Returns:
        Dict containing mean_reward, std_reward, and other metrics
    """
    try:
        rewards = np.array(rewards)
        return {
            'mean_reward': calculate_mean(rewards),
            'std_reward': calculate_std(rewards),
            'min_reward': float(np.min(rewards)) if len(rewards) > 0 else 0.0,
            'max_reward': float(np.max(rewards)) if len(rewards) > 0 else 0.0
        }
    except Exception as e:
        logger = setup_logging('calculations')
        logger.error(f"Error calculating reward metrics: {str(e)}")
        return {
            'mean_reward': 0.0,
            'std_reward': 0.0,
            'min_reward': 0.0,
            'max_reward': 0.0
        }

def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown from returns.
    
    Args:
        returns: Array of returns/rewards
        
    Returns:
        float: Maximum drawdown as a positive number (e.g., 0.20 for 20% drawdown)
    """
    try:
        if len(returns) < 2:
            return 0.0
            
        # Convert to numpy array if not already
        returns = np.array(returns, dtype=float)
        
        # Check for invalid values
        if np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
            logger = setup_logging('calculations')
            logger.warning("Invalid values in returns array")
            return 1.0  # Maximum drawdown
            
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        
        # Avoid division by zero
        valid_indices = running_max > 0
        if not np.any(valid_indices):
            return 1.0
            
        # Calculate drawdowns and convert to positive numbers
        drawdowns = np.zeros_like(running_max)
        drawdowns[valid_indices] = -(cumulative[valid_indices] - running_max[valid_indices]) / running_max[valid_indices]
        
        # Return maximum drawdown
        max_dd = float(np.max(drawdowns))
        return max_dd if not np.isnan(max_dd) else 1.0
        
    except Exception as e:
        logger = setup_logging('calculations')
        logger.error(f"Error calculating max drawdown: {str(e)}")
        return 1.0  # Return maximum drawdown on error

def calculate_win_rate(returns: np.ndarray) -> float:
    """Calculate win rate from returns.
    
    Args:
        returns: Array of returns/rewards
        
    Returns:
        float: Win rate as a decimal (e.g., 0.65 for 65% win rate)
    """
    try:
        if len(returns) == 0:
            return 0.0
            
        returns = np.array(returns)
        wins = np.sum(returns > 0)
        return float(wins) / len(returns)
        
    except Exception as e:
        logger = setup_logging('calculations')
        logger.error(f"Error calculating win rate: {str(e)}")
        return 0.0

def calculate_volatility(returns: np.ndarray, window: int = None) -> Union[float, np.ndarray]:
    """Calculate volatility (standard deviation) of returns.
    
    Args:
        returns: Array of returns/rewards
        window: Optional rolling window size. If provided, returns rolling volatility
        
    Returns:
        float or np.ndarray: Volatility measure(s)
    """
    try:
        if len(returns) < 2:
            return 0.0
            
        returns = np.array(returns)
        if window is not None:
            # Use pandas for rolling calculations
            return pd.Series(returns).rolling(window=window).std(ddof=1).fillna(0).values
        else:
            return calculate_std(returns)
            
    except Exception as e:
        logger = setup_logging('calculations')
        logger.error(f"Error calculating volatility: {str(e)}")
        return 0.0

def calculate_risk_metrics(returns: np.ndarray, prices: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Calculate comprehensive risk metrics.
    
    Args:
        returns: Array of returns/rewards
        prices: Optional array of prices for price-based metrics
        
    Returns:
        Dict containing various risk metrics
    """
    try:
        metrics = {
            'sharpe_ratio': calculate_sharpe_ratio(returns),
            'max_drawdown': calculate_max_drawdown(returns),
            'volatility': calculate_volatility(returns),
            'win_rate': calculate_win_rate(returns)
        }
        
        if len(returns) > 0:
            metrics.update({
                'max_loss': float(np.min(returns)),
                'max_gain': float(np.max(returns)),
                'avg_gain': float(np.mean(returns[returns > 0])) if np.any(returns > 0) else 0.0,
                'avg_loss': float(np.mean(returns[returns < 0])) if np.any(returns < 0) else 0.0
            })
            
        if prices is not None and len(prices) > 1:
            price_returns = np.diff(prices) / prices[:-1]
            metrics['price_volatility'] = calculate_volatility(price_returns)
            
        return metrics
        
    except Exception as e:
        logger = setup_logging('calculations')
        logger.error(f"Error calculating risk metrics: {str(e)}")
        return {
            'sharpe_ratio': 0.0,
            'max_drawdown': 1.0,
            'volatility': 0.0,
            'win_rate': 0.0,
            'max_loss': 0.0,
            'max_gain': 0.0,
            'avg_gain': 0.0,
            'avg_loss': 0.0,
            'price_volatility': 0.0
        }

def calculate_trade_metrics(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive trade metrics from a DataFrame of trades.
    
    Args:
        trades_df: DataFrame containing trade information
        
    Returns:
        Dict containing various trade metrics
    """
    try:
        if len(trades_df) == 0:
            return {}
            
        # Basic metrics
        metrics = {
            'total_trades': len(trades_df),
            'total_pnl': trades_df['pnl'].sum() if 'pnl' in trades_df.columns else 0.0,
            'win_rate': (trades_df['pnl'] > 0).mean() if 'pnl' in trades_df.columns else 0.0
        }
        
        # Add holding time metrics if available
        if 'holding_time' in trades_df.columns:
            metrics.update({
                'avg_holding_time': trades_df['holding_time'].mean(),
                'max_holding_time': trades_df['holding_time'].max(),
                'min_holding_time': trades_df['holding_time'].min()
            })
            
        # Add type-based metrics if available
        if 'type' in trades_df.columns:
            for trade_type in trades_df['type'].unique():
                type_df = trades_df[trades_df['type'] == trade_type]
                metrics[f'{trade_type}_trades'] = len(type_df)
                if 'pnl' in trades_df.columns:
                    metrics[f'{trade_type}_win_rate'] = (type_df['pnl'] > 0).mean()
                    metrics[f'{trade_type}_avg_pnl'] = type_df['pnl'].mean()
                    
        # Add time-based metrics if entry_time is available
        if 'entry_time' in trades_df.columns:
            trades_df['hour'] = pd.to_datetime(trades_df['entry_time']).dt.hour
            for period, hours in [('morning', range(9, 12)), ('afternoon', range(12, 16))]:
                period_df = trades_df[trades_df['hour'].isin(hours)]
                metrics[f'{period}_trades'] = len(period_df)
                if 'pnl' in trades_df.columns:
                    metrics[f'{period}_win_rate'] = (period_df['pnl'] > 0).mean()
                    
        return metrics
        
    except Exception as e:
        logger = setup_logging('calculations')
        logger.error(f"Error calculating trade metrics: {str(e)}")
        return {}

def ensure_project_directories() -> None:
    """Ensure all project directories exist."""
    directories = [
        PROCESSED_DATA_DIR,
        UNPROCESSED_DATA_DIR,
        MODEL_DIR,
        LOGS_DIR,
        OPTUNA_DIR,
        DEPLOYMENT_DIR,
        TRADE_HISTORY_DIR,
        PERFORMANCE_DIR
    ]
    for directory in directories:
        ensure_directory_exists(directory)
        logging.getLogger(__name__).info(f"Ensured directory exists: {directory}")

def create_env(data_path: str, is_eval: bool = False) -> VecNormalize:
    """Create and configure the trading environment.
    
    Args:
        data_path: Path to the data file
        is_eval: Whether this is an evaluation environment
        
    Returns:
        VecNormalize: The configured environment
    """
    try:
        # Import here to avoid circular imports
        from scripts.trading_env import TradingEnv
        
        # Create base environment
        base_env = TradingEnv(data=data_path)
        
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
        logging.getLogger(__name__).error(f"Error creating environment: {str(e)}")
        raise

def evaluate_episode(env, model, episode_num: Optional[int] = None) -> Tuple[Dict[str, float], List[Dict]]:
    """Evaluate a single episode.
    
    Args:
        env: The environment to evaluate in
        model: The model to evaluate
        episode_num: Optional episode number for tracking
        
    Returns:
        Tuple containing metrics dictionary and list of trades
    """
    episode_returns = []
    trades_list = []
    
    obs = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_returns.append(reward)
        
        if info.get('trade'):
            trades_list.append(info['trade'])
    
    # Calculate comprehensive metrics
    metrics = calculate_reward_metrics(episode_returns)
    risk_metrics = calculate_risk_metrics(episode_returns)
    metrics.update(risk_metrics)
    
    if episode_num is not None:
        metrics['episode'] = episode_num
    
    return metrics, trades_list

def load_model(model_path: str, vec_normalize_path: str, train_env) -> Tuple[Optional[PPO], VecNormalize]:
    """Load a saved model and its normalization statistics.
    
    Args:
        model_path: Path to the saved model
        vec_normalize_path: Path to the saved normalization statistics
        train_env: The training environment
        
    Returns:
        Tuple of (loaded model, environment)
    """
    try:
        # Check for both .zip extension and without extension
        if not os.path.exists(model_path) and not model_path.endswith('.zip'):
            model_path_with_ext = f"{model_path}.zip"
            if os.path.exists(model_path_with_ext):
                model_path = model_path_with_ext
        
        if not os.path.exists(model_path):
            return None, train_env
            
        model = PPO.load(model_path, env=train_env)
        
        if os.path.exists(vec_normalize_path):
            train_env = VecNormalize.load(vec_normalize_path, train_env)
            train_env.training = True
            
        return model, train_env
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Error loading model: {str(e)}")
        return None, train_env

def cleanup_resources(temp_files: List[str] = None, environments: List[Any] = None) -> None:
    """Clean up temporary files and environments.
    
    Args:
        temp_files: List of temporary file paths to remove
        environments: List of environments to close
    """
    if temp_files:
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to remove temporary file {file_path}: {e}")

    if environments:
        for env in environments:
            try:
                env.close()
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to close environment: {e}") 