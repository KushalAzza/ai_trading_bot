import os
import gymnasium as gym
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from gymnasium import spaces
from scripts.utils import setup_logging

logger = setup_logging('trading_env')

class TradingEnv(gym.Env):
    """Custom Trading Environment that follows gymnasium interface"""
    
    def __init__(self, data, env_params):
        super(TradingEnv, self).__init__()
        
        try:
            # If data is a string (file path), load it as a DataFrame
            if isinstance(data, str):
                self.df = pd.read_csv(data)
            # If data is already a DataFrame, use it directly
            elif isinstance(data, pd.DataFrame):
                self.df = data.copy()
            else:
                raise ValueError("data must be either a file path or a pandas DataFrame")
            
            # Convert timestamp to datetime
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df['expiry_date'] = pd.to_datetime(self.df['expiry_date'])
            
            # Initialize environment parameters
            self.initial_balance = env_params.get('initial_balance', 100000)
            self.transaction_cost = env_params.get('transaction_cost', 0.0)
            self.max_position = env_params.get('max_position', 1)
            self.stop_loss_pct = env_params.get('stop_loss_pct', 0.02)
            self.max_drawdown_pct = env_params.get('max_drawdown_pct', 0.1)
            
            # Set up feature columns (exclude non-numeric columns)
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns
            self.feature_columns = [col for col in numeric_columns if col not in ['timestamp', 'expiry_date']]
            
            # Define action and observation spaces
            self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
            self.observation_space = spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(len(self.feature_columns),), 
                dtype=np.float32
            )
            
            # Initialize state variables
            self.reset()
            
        except Exception as e:
            if isinstance(data, str):
                raise ValueError(f"Failed to load data from {data}: {str(e)}")
            else:
                raise ValueError(f"Failed to initialize environment: {str(e)}")
    
    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state."""
        super().reset(seed=seed)  # Initialize RNG if provided
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.trades = 0
        self.max_balance = self.initial_balance
        self.drawdown = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get the current observation."""
        return self.df.iloc[self.current_step][self.feature_columns].values.astype(np.float32)
    
    def _calculate_reward(self, action):
        """Calculate the reward for the current step."""
        # Get current and next prices
        current_price = self.df.iloc[self.current_step]['option_ltp']
        next_price = self.df.iloc[self.current_step + 1]['option_ltp'] if self.current_step + 1 < len(self.df) else current_price
        
        # Calculate price change
        price_change = next_price - current_price
        
        # Initialize reward
        reward = 0
        
        # Calculate reward based on action and price change
        if action == 1:  # Buy
            reward = price_change if self.position >= 0 else -price_change
        elif action == 2:  # Sell
            reward = -price_change if self.position <= 0 else price_change
        
        # Penalize for transaction costs
        if action != 0:
            reward -= self.transaction_cost
        
        return reward
    
    def step(self, action):
        """Execute one time step within the environment."""
        # Check if episode is done
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, False, {}
        
        # Get current price
        current_price = self.df.iloc[self.current_step]['option_ltp']
        
        # Execute action
        reward = self._calculate_reward(action)
        
        # Update position based on action
        if action == 1:  # Buy
            if self.position < self.max_position:
                self.position += 1
                self.trades += 1
                self.balance -= current_price
        elif action == 2:  # Sell
            if self.position > -self.max_position:
                self.position -= 1
                self.trades += 1
                self.balance += current_price
        
        # Update maximum balance
        self.max_balance = max(self.max_balance, self.balance)
        
        # Calculate drawdown
        self.drawdown = (self.max_balance - self.balance) / self.max_balance
        
        # Move to next step
        self.current_step += 1
        
        # Get next observation
        obs = self._get_observation()
        
        # Check if stop loss or max drawdown is hit
        stop_loss_hit = self.drawdown >= self.stop_loss_pct
        max_drawdown_hit = self.drawdown >= self.max_drawdown_pct
        
        # Episode is done if we hit stop loss or max drawdown
        done = stop_loss_hit or max_drawdown_hit or self.current_step >= len(self.df) - 1
        
        # Prepare info dictionary
        info = {
            'balance': self.balance,
            'position': self.position,
            'trades': self.trades,
            'drawdown': self.drawdown,
            'stop_loss_hit': stop_loss_hit,
            'max_drawdown_hit': max_drawdown_hit
        }
        
        # For new Gym API compatibility
        terminated = done
        truncated = False
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Render the environment."""
        pass
    
    def close(self):
        """Clean up the environment."""
        pass
