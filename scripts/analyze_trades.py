import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from scripts.utils import (
    setup_logging,
    ensure_directory_exists,
    LOGS_DIR
)

logger = setup_logging('analyze_trades')

def load_trades() -> pd.DataFrame:
    """Load and preprocess trade data."""
    try:
        trades_file = f"{LOGS_DIR}/paper_trades.csv"
        df = pd.read_csv(trades_file)
        
        # Convert timestamps to datetime
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        
        return df
        
    except FileNotFoundError:
        logger.error(f"Trades file not found: {trades_file}")
        raise
    except Exception as e:
        logger.error(f"Error loading trades: {str(e)}")
        raise

def calculate_basic_metrics(df: pd.DataFrame) -> Dict:
    """Calculate basic trading metrics."""
    try:
        metrics = {
            'total_trades': len(df),
            'win_trades': len(df[df['pnl'] > 0]),
            'loss_trades': len(df[df['pnl'] <= 0]),
            'total_pnl': df['pnl'].sum(),
            'avg_pnl': df['pnl'].mean(),
            'median_pnl': df['pnl'].median(),
            'std_pnl': df['pnl'].std(),
            'max_profit': df['pnl'].max(),
            'max_loss': df['pnl'].min(),
            'win_rate': len(df[df['pnl'] > 0]) / len(df) if len(df) > 0 else 0
        }
        
        # Calculate holding times
        df['holding_time'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60
        metrics.update({
            'avg_holding_time': df['holding_time'].mean(),
            'median_holding_time': df['holding_time'].median()
        })
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating basic metrics: {str(e)}")
        raise

def calculate_risk_metrics(df: pd.DataFrame) -> Dict:
    """Calculate risk-related metrics."""
    try:
        # Group by date for daily metrics
        df['date'] = df['exit_time'].dt.date
        daily_pnl = df.groupby('date')['pnl'].sum()
        
        # Calculate drawdown
        cumulative_pnl = daily_pnl.cumsum()
        rolling_max = cumulative_pnl.expanding().max()
        daily_drawdown = cumulative_pnl - rolling_max
        max_drawdown = daily_drawdown.min()
        
        # Calculate risk metrics
        metrics = {
            'max_drawdown': max_drawdown,
            'profit_factor': abs(df[df['pnl'] > 0]['pnl'].sum() / df[df['pnl'] < 0]['pnl'].sum()) if len(df[df['pnl'] < 0]) > 0 else float('inf'),
            'avg_win_loss_ratio': abs(df[df['pnl'] > 0]['pnl'].mean() / df[df['pnl'] < 0]['pnl'].mean()) if len(df[df['pnl'] < 0]) > 0 else float('inf'),
            'sharpe_ratio': daily_pnl.mean() / daily_pnl.std() if daily_pnl.std() != 0 else 0,
            'daily_win_rate': (daily_pnl > 0).mean()
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {str(e)}")
        raise

def analyze_by_option_type(df: pd.DataFrame) -> Dict:
    """Analyze performance by option type."""
    try:
        type_analysis = {}
        
        for opt_type in ['CE', 'PE']:
            type_df = df[df['type'] == opt_type]
            if len(type_df) > 0:
                type_analysis[opt_type] = {
                    'total_trades': len(type_df),
                    'win_rate': len(type_df[type_df['pnl'] > 0]) / len(type_df),
                    'avg_pnl': type_df['pnl'].mean(),
                    'total_pnl': type_df['pnl'].sum(),
                    'max_profit': type_df['pnl'].max(),
                    'max_loss': type_df['pnl'].min()
                }
                
        return type_analysis
        
    except Exception as e:
        logger.error(f"Error analyzing by option type: {str(e)}")
        raise

def analyze_by_time(df: pd.DataFrame) -> Dict:
    """Analyze performance by time of day."""
    try:
        df['hour'] = df['entry_time'].dt.hour
        time_analysis = {}
        
        # Morning session (9:30-12:00)
        morning = df[(df['hour'] >= 9) & (df['hour'] < 12)]
        # Afternoon session (12:00-15:30)
        afternoon = df[(df['hour'] >= 12) & (df['hour'] <= 15)]
        
        for period, period_df in [('morning', morning), ('afternoon', afternoon)]:
            if len(period_df) > 0:
                time_analysis[period] = {
                    'total_trades': len(period_df),
                    'win_rate': len(period_df[period_df['pnl'] > 0]) / len(period_df),
                    'avg_pnl': period_df['pnl'].mean(),
                    'total_pnl': period_df['pnl'].sum()
                }
                
        return time_analysis
        
    except Exception as e:
        logger.error(f"Error analyzing by time: {str(e)}")
        raise

def analyze_trades() -> None:
    """Main function to analyze trading performance."""
    try:
        logger.info("Starting trade analysis...")
        
        # Load trades
        df = load_trades()
        
        if len(df) == 0:
            logger.warning("No trades found for analysis")
            return
            
        # Calculate metrics
        basic_metrics = calculate_basic_metrics(df)
        risk_metrics = calculate_risk_metrics(df)
        type_analysis = analyze_by_option_type(df)
        time_analysis = analyze_by_time(df)
        
        # Log results
        logger.info("\nBasic Performance Metrics:")
        logger.info(f"Total Trades: {basic_metrics['total_trades']}")
        logger.info(f"Win Rate: {basic_metrics['win_rate']:.2%}")
        logger.info(f"Total P&L: ₹{basic_metrics['total_pnl']:,.2f}")
        logger.info(f"Average P&L per Trade: ₹{basic_metrics['avg_pnl']:,.2f}")
        logger.info(f"Median P&L per Trade: ₹{basic_metrics['median_pnl']:,.2f}")
        logger.info(f"Average Holding Time: {basic_metrics['avg_holding_time']:.1f} minutes")
        
        logger.info("\nRisk Metrics:")
        logger.info(f"Max Drawdown: ₹{risk_metrics['max_drawdown']:,.2f}")
        logger.info(f"Profit Factor: {risk_metrics['profit_factor']:.2f}")
        logger.info(f"Average Win/Loss Ratio: {risk_metrics['avg_win_loss_ratio']:.2f}")
        logger.info(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
        logger.info(f"Daily Win Rate: {risk_metrics['daily_win_rate']:.2%}")
        
        logger.info("\nPerformance by Option Type:")
        for opt_type, metrics in type_analysis.items():
            logger.info(f"\n{opt_type} Options:")
            logger.info(f"Total Trades: {metrics['total_trades']}")
            logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
            logger.info(f"Average P&L: ₹{metrics['avg_pnl']:,.2f}")
            logger.info(f"Total P&L: ₹{metrics['total_pnl']:,.2f}")
            
        logger.info("\nPerformance by Time of Day:")
        for period, metrics in time_analysis.items():
            logger.info(f"\n{period.capitalize()} Session:")
            logger.info(f"Total Trades: {metrics['total_trades']}")
            logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
            logger.info(f"Average P&L: ₹{metrics['avg_pnl']:,.2f}")
            
        # Generate warnings for potential issues
        if basic_metrics['win_rate'] < 0.4:
            logger.warning("\n⚠️ Warning: Win rate is below 40%")
        if risk_metrics['profit_factor'] < 1.5:
            logger.warning("⚠️ Warning: Profit factor is below 1.5")
        if risk_metrics['max_drawdown'] < -5000:
            logger.warning("⚠️ Warning: Large drawdown detected")
            
    except Exception as e:
        logger.error(f"Error in trade analysis: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        ensure_directory_exists(LOGS_DIR)
        analyze_trades()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        raise