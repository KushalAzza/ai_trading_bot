import os
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from scripts.trading_env import TradingEnv
from scripts.utils import (
    setup_logging,
    ensure_directory_exists,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    LOGS_DIR,
    DEPLOYMENT_DIR,
    TRADE_HISTORY_DIR,
    PERFORMANCE_DIR,
    TRANSACTION_COST,
    INITIAL_BALANCE,
    BASE_LOT_SIZE,
    MAX_EXPOSURE,
    DAILY_RISK_LIMIT,
    OPTUNA_DIR
)

# Set up logging
logger = setup_logging('deploy_model')

# Create deployment directories
for directory in [DEPLOYMENT_DIR, TRADE_HISTORY_DIR, PERFORMANCE_DIR]:
    ensure_directory_exists(directory)
    logger.info(f"Ensured directory exists: {directory}")

class ModelDeployer:
    """Class to handle model deployment and live trading"""
    
    def __init__(self):
        """Initialize the model deployer"""
        try:
            # Load model and environment
            self.model_path = os.path.join(MODEL_DIR, 'final_model.zip')
            self.vec_normalize_path = os.path.join(MODEL_DIR, 'vec_normalize.pkl')
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            # Load environment parameters from training
            self.env_params = self._load_env_params()
            
            # Create dummy environment for prediction
            self.env = self._create_environment()
            
            # Load the trained model
            self.model = PPO.load(self.model_path)
            logger.info(f"Loaded model from {self.model_path}")
            
            # Initialize trading state using environment parameters
            self.initial_balance = INITIAL_BALANCE
            self.balance = self.initial_balance
            self.entry_price = 0.0
            self.position_size = 0
            self.daily_trades = 0
            self.lot_size = BASE_LOT_SIZE
            self.transaction_cost = TRANSACTION_COST
            
            # Use parameters from training environment
            self.max_position = self.env_params.get('max_position', 1)
            self.stop_loss_pct = self.env_params.get('stop_loss_pct', 0.02)
            self.max_drawdown_pct = self.env_params.get('max_drawdown_pct', 0.1)
            self.max_daily_trades = self.env_params.get('max_trades_per_day', 5)
            self.max_balance = self.initial_balance
            self.current_date = datetime.now().date()
            
            # Initialize trade history
            self.trade_history = []
            self.daily_stats = {
                'date': self.current_date,
                'trades': 0,
                'profit': 0.0,
                'max_drawdown': 0.0,
                'win_trades': 0,
                'lose_trades': 0
            }
            
            # Safety parameters
            self.max_consecutive_losses = 3
            self.consecutive_losses = 0
            self.safety_mode = False
            self.safety_cooldown_minutes = 30
            self.safety_triggered_time = None
            
            logger.info("Model deployer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model deployer: {str(e)}")
            raise
    
    def _load_env_params(self) -> Dict[str, Any]:
        """Load environment parameters from training"""
        try:
            env_params_path = os.path.join(MODEL_DIR, 'env_params.json')
            if not os.path.exists(env_params_path):
                raise FileNotFoundError(f"Environment parameters not found at {env_params_path}")
                
            with open(env_params_path, 'r') as f:
                params = json.load(f)
            logger.info(f"Loaded environment parameters from {env_params_path}")
            return params
            
        except Exception as e:
            logger.error(f"Error loading environment parameters: {str(e)}")
            raise ValueError("Cannot deploy model without training parameters")
    
    def _create_environment(self) -> VecNormalize:
        """Create environment for prediction"""
        try:
            # Create a dummy environment with the same parameters as training
            dummy_env = DummyVecEnv([lambda: TradingEnv(
                data_path=os.path.join(PROCESSED_DATA_DIR, 'training_data.csv'),
                **self.env_params
            )])
            
            # Load normalization statistics if available
            if os.path.exists(self.vec_normalize_path):
                env = VecNormalize.load(self.vec_normalize_path, dummy_env)
                env.training = False  # Disable training mode
                env.norm_reward = False  # Don't normalize rewards during deployment
                logger.info(f"Loaded normalization statistics from {self.vec_normalize_path}")
            else:
                env = VecNormalize(
                    dummy_env,
                    norm_obs=True,
                    norm_reward=False,
                    clip_obs=10.0,
                    clip_reward=10.0
                )
                logger.warning("Normalization statistics not found, using default values")
            
            return env
            
        except Exception as e:
            logger.error(f"Error creating environment: {str(e)}")
            raise
    
    def _reset_daily_stats(self) -> None:
        """Reset daily trading statistics"""
        # Save previous day's stats if there were trades
        if self.daily_stats['trades'] > 0:
            self._save_daily_stats()
            
        # Reset stats for new day
        self.daily_stats = {
            'date': datetime.now().date(),
            'trades': 0,
            'profit': 0.0,
            'max_drawdown': 0.0,
            'win_trades': 0,
            'lose_trades': 0
        }
        self.daily_trades = 0
        logger.info(f"Reset daily stats for {self.daily_stats['date']}")
    
    def _save_daily_stats(self) -> None:
        """Save daily trading statistics"""
        try:
            # Create filename with date
            filename = f"daily_stats_{self.daily_stats['date']}.json"
            filepath = os.path.join(PERFORMANCE_DIR, filename)
            
            # Add additional metrics
            self.daily_stats['balance'] = self.balance
            self.daily_stats['return_pct'] = (self.balance / self.initial_balance - 1) * 100
            self.daily_stats['win_rate'] = (self.daily_stats['win_trades'] / self.daily_stats['trades'] * 100) if self.daily_stats['trades'] > 0 else 0
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(self.daily_stats, f, indent=4, default=str)
                
            logger.info(f"Saved daily stats to {filepath}")
        except Exception as e:
            logger.error(f"Error saving daily stats: {str(e)}")
    
    def _can_open_new_trade(self, order_size: float) -> Tuple[bool, str]:
        """Check if a new trade can be opened based on various conditions"""
        # Check if in safety mode
        if self.safety_mode:
            current_time = datetime.now()
            if (current_time - self.safety_triggered_time).total_seconds() / 60 < self.safety_cooldown_minutes:
                remaining_minutes = self.safety_cooldown_minutes - int((current_time - self.safety_triggered_time).total_seconds() / 60)
                return False, f"Safety mode active for {remaining_minutes} more minutes"
            else:
                self.safety_mode = False
                self.consecutive_losses = 0
                logger.info("Safety mode deactivated")
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return False, f"Daily trade limit reached ({self.max_daily_trades})"
        
        # Check if already in a position
        if self.position_size > 0:
            return False, "Already in a position"
        
        # Check if order size is affordable
        if order_size * self.lot_size > self.balance:
            return False, f"Insufficient balance for order size {order_size * self.lot_size}"
        
        # Check if balance is below minimum threshold (20% of initial)
        if self.balance < self.initial_balance * 0.2:
            return False, f"Balance too low: {self.balance:.2f}"
        
        # All checks passed
        return True, "Trade allowed"
    
    def _check_safety_conditions(self, profit: float) -> None:
        """Check safety conditions and update safety mode if needed"""
        if profit < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.safety_mode = True
                self.safety_triggered_time = datetime.now()
                logger.warning(f"Safety mode activated after {self.consecutive_losses} consecutive losses")
        else:
            self.consecutive_losses = 0
    
    def execute_trades(self, df_live: pd.DataFrame) -> None:
        """Execute trades based on model predictions"""
        try:
            # Check if a new day has started
            current_date = datetime.now().date()
            if current_date != self.current_date:
                self._reset_daily_stats()
                self.current_date = current_date
            
            # Get the latest data point and ensure timestamp is datetime
            latest_data = df_live.iloc[-1]
            if 'timestamp' in latest_data and not pd.api.types.is_datetime64_any_dtype(latest_data['timestamp']):
                latest_data['timestamp'] = pd.to_datetime(latest_data['timestamp'])
            
            # Create observation and get prediction
            obs = self._prepare_observation(latest_data)
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Process action
            self._process_action(action, latest_data)
            
            # Log state
            state = {
                'timestamp': datetime.now(),
                'balance': self.balance,
                'position_size': self.position_size,
                'entry_price': self.entry_price,
                'daily_trades': self.daily_trades,
                'max_daily_trades': self.max_daily_trades,
                'safety_mode': self.safety_mode,
                'consecutive_losses': self.consecutive_losses,
                'return_pct': (self.balance / self.initial_balance - 1) * 100
            }
            logger.debug(f"Current state: {state}")
            
        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}")
    
    def _prepare_observation(self, data: pd.Series) -> np.ndarray:
        """Prepare observation for model prediction"""
        try:
            # Reset the environment with the new data
            dummy_env = self.env.venv.envs[0].env
            
            # Extract features in the same order as during training
            feature_columns = dummy_env.feature_columns
            features = np.array([data[col] for col in feature_columns if col in data], dtype=np.float32)
            
            # Normalize the observation
            obs = self.env.normalize_obs(features.reshape(1, -1))
            
            return obs
            
        except Exception as e:
            logger.error(f"Error preparing observation: {str(e)}")
            # Return zeros as fallback
            return np.zeros((1, len(self.env.venv.envs[0].env.feature_columns)), dtype=np.float32)
    
    def _process_action(self, action: int, data: pd.Series) -> None:
        """Process the action from the model"""
        try:
            timestamp = data.get('timestamp', datetime.now())
            option_ltp = data.get('option_ltp', 0.0)
            
            # Log the action
            action_str = "HOLD" if action == 0 else "BUY" if action == 1 else "SELL"
            logger.info(f"Action: {action_str}, Price: {option_ltp:.2f}")
            
            # Process based on action
            if action == 1:  # BUY
                # Check if we can open a new trade
                can_trade, reason = self._can_open_new_trade(1)
                if not can_trade:
                    logger.info(f"Cannot open BUY trade: {reason}")
                    return
                
                # Execute BUY
                self.position_size = 1
                self.entry_price = option_ltp
                self.daily_trades += 1
                self.daily_stats['trades'] += 1
                
                # Record trade
                trade = {
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'price': option_ltp,
                    'size': self.position_size * self.lot_size,
                    'balance_before': self.balance,
                    'cost': self.transaction_cost
                }
                self.trade_history.append(trade)
                
                logger.info(f"BUY executed at {option_ltp:.2f}, Size: {self.position_size * self.lot_size}")
                
            elif action == 2:  # SELL
                # Check if we have a position to sell
                if self.position_size == 0:
                    logger.info("Cannot SELL: No position to sell")
                    return
                
                # Calculate profit/loss
                price_diff = option_ltp - self.entry_price
                profit = price_diff * self.position_size * self.lot_size - self.transaction_cost
                self.balance += profit
                
                # Update max balance
                self.max_balance = max(self.max_balance, self.balance)
                
                # Update daily stats
                self.daily_stats['profit'] += profit
                if profit > 0:
                    self.daily_stats['win_trades'] += 1
                else:
                    self.daily_stats['lose_trades'] += 1
                
                # Calculate drawdown
                drawdown = (self.max_balance - self.balance) / self.max_balance if self.max_balance > 0 else 0
                self.daily_stats['max_drawdown'] = max(self.daily_stats['max_drawdown'], drawdown)
                
                # Record trade
                trade = {
                    'timestamp': timestamp,
                    'action': 'SELL',
                    'price': option_ltp,
                    'size': self.position_size * self.lot_size,
                    'entry_price': self.entry_price,
                    'profit': profit,
                    'balance_after': self.balance,
                    'drawdown': drawdown
                }
                self.trade_history.append(trade)
                
                # Check safety conditions
                self._check_safety_conditions(profit)
                
                # Reset position
                self.position_size = 0
                self.entry_price = 0
                
                logger.info(f"SELL executed at {option_ltp:.2f}, Profit: {profit:.2f}, Balance: {self.balance:.2f}")
                
            # Save trade history periodically
            if len(self.trade_history) % 10 == 0:
                self._save_trade_history()
                
        except Exception as e:
            logger.error(f"Error processing action: {str(e)}")
    
    def _save_trade_history(self) -> None:
        """Save trade history to file"""
        try:
            if not self.trade_history:
                return
                
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trade_history_{timestamp}.csv"
            filepath = os.path.join(TRADE_HISTORY_DIR, filename)
            
            # Save to CSV
            pd.DataFrame(self.trade_history).to_csv(filepath, index=False)
            logger.info(f"Saved trade history to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving trade history: {str(e)}")
    
    def close_all_positions(self) -> None:
        """Close all open positions"""
        try:
            if self.position_size == 0:
                logger.info("No positions to close")
                return
            
            # Get latest price (this would be replaced with actual market data)
            # For simulation, we'll use the last entry price
            latest_price = self.entry_price
            
            # Calculate profit/loss
            price_diff = latest_price - self.entry_price
            profit = price_diff * self.position_size * self.lot_size - self.transaction_cost
            self.balance += profit
            
            # Update max balance
            self.max_balance = max(self.max_balance, self.balance)
            
            # Update daily stats
            self.daily_stats['profit'] += profit
            if profit > 0:
                self.daily_stats['win_trades'] += 1
            else:
                self.daily_stats['lose_trades'] += 1
            
            # Calculate drawdown
            drawdown = (self.max_balance - self.balance) / self.max_balance if self.max_balance > 0 else 0
            self.daily_stats['max_drawdown'] = max(self.daily_stats['max_drawdown'], drawdown)
            
            # Record trade
            trade = {
                'timestamp': datetime.now(),
                'action': 'FORCE_CLOSE',
                'price': latest_price,
                'size': self.position_size * self.lot_size,
                'entry_price': self.entry_price,
                'profit': profit,
                'balance_after': self.balance,
                'drawdown': drawdown
            }
            self.trade_history.append(trade)
            
            # Reset position
            self.position_size = 0
            self.entry_price = 0
            
            logger.info(f"Forced close at {latest_price:.2f}, Profit: {profit:.2f}, Balance: {self.balance:.2f}")
            
            # Save trade history
            self._save_trade_history()
            
            # Save daily stats
            self._save_daily_stats()
            
        except Exception as e:
            logger.error(f"Error closing positions: {str(e)}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of trading performance"""
        try:
            if not self.trade_history:
                return {
                    'status': 'No trades executed',
                    'balance': self.balance,
                    'return_pct': 0.0
                }
            
            # Calculate performance metrics
            total_trades = len([t for t in self.trade_history if t.get('action') in ['BUY', 'SELL']])
            win_trades = len([t for t in self.trade_history if t.get('profit', 0) > 0])
            lose_trades = len([t for t in self.trade_history if t.get('profit', 0) <= 0])
            win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate profit metrics
            total_profit = sum([t.get('profit', 0) for t in self.trade_history if 'profit' in t])
            avg_profit = total_profit / win_trades if win_trades > 0 else 0
            avg_loss = sum([t.get('profit', 0) for t in self.trade_history if t.get('profit', 0) <= 0]) / lose_trades if lose_trades > 0 else 0
            
            # Calculate risk metrics
            max_drawdown = max([t.get('drawdown', 0) for t in self.trade_history]) if self.trade_history else 0
            
            return {
                'status': 'active',
                'initial_balance': self.initial_balance,
                'current_balance': self.balance,
                'return_pct': (self.balance / self.initial_balance - 1) * 100,
                'total_trades': total_trades,
                'win_trades': win_trades,
                'lose_trades': lose_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf'),
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100,
                'safety_mode': self.safety_mode,
                'consecutive_losses': self.consecutive_losses
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'balance': self.balance
            }

def main():
    """Main function to run the model deployment"""
    try:
        logger.info("Starting model deployment")
        
        # Initialize deployer
        deployer = ModelDeployer()
        
        # Load training data for simulation
        df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'training_data.csv'))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Process data in chunks
        chunk_size = 10
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            deployer.execute_trades(chunk)
            time.sleep(0.1)  # Simulate time delay
        
        # Close all positions and get final summary
        deployer.close_all_positions()
        summary = deployer.get_performance_summary()
        logger.info("Final performance summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("Model deployment completed")
        
    except Exception as e:
        logger.error(f"Error in model deployment: {str(e)}")
        raise

if __name__ == "__main__":
    main()