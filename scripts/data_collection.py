import os
import sys
import time
import json
import requests
from datetime import datetime, timedelta, date, time as datetime_time
import pandas as pd
import numpy as np
import schedule
import talib
from dhanhq import dhanhq
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional, Any, Union

# Add project root to Python path if running directly from scripts directory
if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == 'scripts':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)

from scripts.utils import (
    setup_logging,
    ensure_directory_exists,
    safe_float,
    safe_int,
    safe_get,
    PROCESSED_DATA_DIR,
    UNPROCESSED_DATA_DIR,
    MODEL_DIR,
    LOGS_DIR
)

# Load environment variables
load_dotenv()

# Set up logging
logger = setup_logging('data_collection')

# Create required directories
for directory in [LOGS_DIR, MODEL_DIR, PROCESSED_DATA_DIR, UNPROCESSED_DATA_DIR]:
    ensure_directory_exists(directory)
logger.debug(f"Required directories verified: {', '.join([LOGS_DIR, MODEL_DIR, PROCESSED_DATA_DIR, UNPROCESSED_DATA_DIR])}")

class NiftyOptionChainCollector:
    """Class for collecting and processing Nifty option chain data."""
    
    def __init__(self):
        """Initialize the collector with configuration and constants."""
        try:
            # Initialize Telegram configuration
            self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            # Initialize Dhan client
            self.client_id = os.getenv('DHAN_CLIENT_ID')
            self.access_token = os.getenv('DHAN_ACCESS_TOKEN')
            self.access_token_expiry = os.getenv('DHAN_ACCESS_TOKEN_EXPIRY')
            
            # Set up directory paths
            self.PROCESSED_DATA_DIR = PROCESSED_DATA_DIR
            self.UNPROCESSED_DATA_DIR = UNPROCESSED_DATA_DIR
            self.LOGS_DIR = LOGS_DIR
            self.MODEL_DIR = MODEL_DIR
            
            # Set up constants
            self.setup_constants()
            
            # Initialize Dhan client
            logger.info(f"Initializing collector [client_id: {self.client_id}, token: {'Present' if self.access_token else 'Missing'}]")
            self.dhan = dhanhq(client_id=self.client_id, access_token=self.access_token)
            
            # Initialize state
            self._current_time = None
            self._cached_data = {}
            
        except Exception as e:
            logger.error(f"Error initializing collector: {str(e)}")
            raise
    
    def setup_constants(self):
        """Set up class constants and configuration."""
        # Class constants
        self.REQUIRED_COLUMNS = ['sma', 'macd', 'macd_signal', 'macd_histogram', 'option_rsi', 'nifty_rsi', 'atr']
        self.DEFAULT_FLOAT_PRECISION = 2
        self.GREEK_PRECISION = 5
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 20  # seconds
        
        # Trading holidays for 2025
        self.TRADING_HOLIDAYS_2025 = {
            date(2025, 2, 26),  # Mahashivratri
            date(2025, 3, 14),  # Holi
            date(2025, 3, 31),  # Id-Ul-Fitr (Ramadan Eid)
            date(2025, 4, 10),  # Shri Mahavir Jayanti
            date(2025, 4, 14),  # Dr. Baba Saheb Ambedkar Jayanti
            date(2025, 4, 18),  # Good Friday
            date(2025, 5, 1),   # Maharashtra Day
            date(2025, 8, 15),  # Independence Day / Parsi New Year
            date(2025, 8, 27),  # Shri Ganesh Chaturthi
            date(2025, 10, 2),  # Mahatma Gandhi Jayanti/Dussehra
            date(2025, 10, 21), # Diwali Laxmi Pujan
            date(2025, 10, 22), # Balipratipada
            date(2025, 11, 5),  # Prakash Gurpurb Sri Guru Nanak Dev
            date(2025, 12, 25)  # Christmas
        }
        
        # Constants
        self.NIFTY_SCRIP_ID = 13  # NIFTY scrip ID
        self.VIX_SCRIP_ID = 21  # VIX scrip ID
        self.BANKNIFTY_SCRIP_ID = 25  # BANKNIFTY scrip ID
        self.NUM_STRIKES = 4  # Number of strikes above and below ATM
        self.HISTORICAL_DATA_ROWS = 8000  # Enough for all indicators
        
        # Technical indicator parameters
        self.SMA_PERIOD = 5  # 5 polls = 25 minutes
        self.RSI_PERIOD = 14  # 14 polls = 70 minutes
        self.MACD_FAST = 12  # 12 polls = 60 minutes
        self.MACD_SLOW = 26  # 26 polls = 130 minutes
        self.MACD_SIGNAL = 9  # 9 polls = 45 minutes

    # Utility methods for data type conversion and validation
    def safe_float(self, value: Any, default: float = 0.0, precision: int = None) -> float:
        """Safely convert value to float with optional precision"""
        try:
            result = float(value) if value is not None else default
            return round(result, precision if precision is not None else self.DEFAULT_FLOAT_PRECISION)
        except (ValueError, TypeError):
            return default

    def safe_int(self, value: Any, default: int = 0) -> int:
        """Safely convert value to int"""
        try:
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    def safe_get(self, data: Dict, path: List[str], default: Any = None) -> Any:
        """Safely navigate nested dictionaries"""
        for key in path:
            if not isinstance(data, dict):
                return default
            data = data.get(key, default)
            if data is None:
                return default
        return data

    def validate_api_response(self, response: Dict, error_context: str) -> None:
        """Validate API response format and status"""
        if not isinstance(response, dict):
            raise Exception(f"{error_context}: Invalid response type: {type(response)}")
        if response.get('status') != 'success':
            raise Exception(f"{error_context}: {response.get('remarks', 'Unknown error')}")

    def ensure_dataframe_columns(self, df: pd.DataFrame, columns: List[str], default_value: Any = 0.0) -> pd.DataFrame:
        """Ensure DataFrame has required columns"""
        for col in columns:
            if col not in df.columns:
                df[col] = default_value
        return df

    def process_option_data(self, option_data: Dict) -> Tuple[float, float, float, int]:
        """Process common option data fields"""
        last_price = safe_float(option_data.get('last_price'))
        ask_price = safe_float(option_data.get('top_ask_price'))
        bid_price = safe_float(option_data.get('top_bid_price'))
        volume = safe_int(option_data.get('volume'))
        return last_price, ask_price, bid_price, volume

    # Properties
    @property
    def current_time(self) -> datetime:
        """Get current timestamp, reuse if already set"""
        if self._current_time is None:
            self._current_time = datetime.now()
        return self._current_time

    @property
    def today(self) -> date:
        """Get current date"""
        return self.current_time.date()

    @property
    def current_time_only(self) -> datetime_time:
        """Get current time without date"""
        return self.current_time.time()

    @property
    def current_timestamp(self) -> str:
        """Get formatted current timestamp"""
        return self.current_time.strftime('%Y-%m-%d %H:%M:%S')

    @property
    def historical_file(self) -> str:
        """Get historical data file path"""
        return f'{self.UNPROCESSED_DATA_DIR}/historical_data.csv'

    @property
    def processed_file(self) -> str:
        """Get processed data file path"""
        return f"{self.PROCESSED_DATA_DIR}/processed_data.csv"

    @property
    def is_trading_day(self) -> bool:
        """Check if today is a trading day"""
        return self.today.weekday() < 5 and self.today not in self.TRADING_HOLIDAYS_2025

    def check_token_expiry(self):
        """Check if the access token is nearing expiry"""
        try:
            if not self.access_token_expiry:
                logger.warning("Access token expiry date not set in .env file")
                return
                
            expiry_date = datetime.strptime(self.access_token_expiry, '%Y-%m-%d').date()
            days_until_expiry = (expiry_date - self.today).days
            
            if days_until_expiry <= 3 and self.is_trading_day:
                # Check if current time is one of the alert times
                alert_times = [
                    datetime.strptime('09:30:00', '%H:%M:%S').time(),
                    datetime.strptime('12:30:00', '%H:%M:%S').time(),
                    datetime.strptime('15:00:00', '%H:%M:%S').time()
                ]
                
                # Allow a 5-minute window for each alert time
                for alert_time in alert_times:
                    alert_datetime = datetime.combine(self.today, alert_time)
                    current_datetime = datetime.combine(self.today, self.current_time_only)
                    time_diff = abs((current_datetime - alert_datetime).total_seconds() / 60)
                    
                    if time_diff <= 2.5:  # Within 2.5 minutes before or after alert time
                        warning_message = (
                            f"⚠️ <b>Access Token Expiry Warning</b>\n\n"
                            f"Your access token will expire in {days_until_expiry} days on {self.access_token_expiry}.\n"
                            f"Please update your token before it expires.\n\n"
                            f"Current time: {self.current_time_only.strftime('%H:%M:%S')}"
                        )
                        logger.warning(f"Access token expiring in {days_until_expiry} days")
                        self.send_telegram_message(warning_message)
                        break
                
        except Exception as e:
            logger.error(f"Error checking token expiry: {str(e)}")

    def get_current_expiry(self) -> str:
        """Get the nearest expiry date from the expiry list.
        
        Returns:
            str: Current expiry date in YYYY-MM-DD format
        """
        try:
            response = self.dhan.expiry_list(
                under_security_id=self.NIFTY_SCRIP_ID,
                under_exchange_segment="IDX_I"
            )
            
            if not isinstance(response, dict):
                raise Exception(f"Invalid response type: {type(response)}")
                
            if response.get('status') != 'success':
                raise Exception(f"API returned error status: {response.get('status')}")
                
            expiry_data = response.get('data', {}).get('data', [])
            if not expiry_data:
                raise Exception("No expiry dates in response")
                
            expiry_dates = []
            if isinstance(expiry_data, list):
                expiry_dates = expiry_data
            elif isinstance(expiry_data, dict):
                expiry_dates = [expiry_data.get('expiry')] if expiry_data.get('expiry') else []
            
            if not expiry_dates:
                raise Exception("No valid expiry dates found in response")
            
            parsed_dates = []
            for date_str in expiry_dates:
                try:
                    for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%Y%m%d']:
                        try:
                            parsed_date = datetime.strptime(str(date_str), fmt).date()
                            parsed_dates.append(parsed_date)
                            break
                        except ValueError:
                            continue
                except Exception:
                    continue
            
            if not parsed_dates:
                raise Exception("No valid dates could be parsed from the response")
            
            today = self.today
            # Filter out past dates and sort remaining dates
            valid_expiries = sorted([exp for exp in parsed_dates if exp >= today])
            
            if not valid_expiries:
                raise Exception("No valid expiry dates found")
            
            return valid_expiries[0].strftime('%Y-%m-%d')
            
        except Exception as e:
            logger.error(f"Error getting current expiry: {str(e)}")
            raise

    def get_next_expiry(self) -> str:
        """Get the next available expiry date after the current expiry.
        
        Returns:
            str: Next expiry date in YYYY-MM-DD format
        """
        try:
            response = self.dhan.expiry_list(
                under_security_id=self.NIFTY_SCRIP_ID,
                under_exchange_segment="IDX_I"
            )
            
            if not isinstance(response, dict):
                raise Exception(f"Invalid response type: {type(response)}")
                
            if response.get('status') != 'success':
                raise Exception(f"API returned error status: {response.get('status')}")
                
            expiry_data = response.get('data', {}).get('data', [])
            if not expiry_data:
                raise Exception("No expiry dates in response")
                
            expiry_dates = []
            if isinstance(expiry_data, list):
                expiry_dates = expiry_data
            elif isinstance(expiry_data, dict):
                expiry_dates = [expiry_data.get('expiry')] if expiry_data.get('expiry') else []
            
            if not expiry_dates:
                raise Exception("No valid expiry dates found in response")
            
            parsed_dates = []
            for date_str in expiry_dates:
                try:
                    for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%Y%m%d']:
                        try:
                            parsed_date = datetime.strptime(str(date_str), fmt).date()
                            parsed_dates.append(parsed_date)
                            break
                        except ValueError:
                            continue
                except Exception:
                    continue
            
            if not parsed_dates:
                raise Exception("No valid dates could be parsed from the response")
            
            today = self.today
            # Filter out past dates and sort remaining dates
            valid_expiries = sorted([exp for exp in parsed_dates if exp >= today])
            
            if len(valid_expiries) < 2:
                raise Exception("No next expiry date available")
            
            # Return the second available expiry
            return valid_expiries[1].strftime('%Y-%m-%d')
            
        except Exception as e:
            logger.error(f"Error getting next expiry: {str(e)}")
            raise

    def get_market_data(self) -> Dict:
        """Fetch market data for indices"""
        try:
            market_data = {}
            market_indices = {
                "IDX_I": [
                    self.NIFTY_SCRIP_ID,
                    self.VIX_SCRIP_ID,
                    self.BANKNIFTY_SCRIP_ID
                ]
            }
            
            indices_data = self.dhan.ohlc_data(market_indices)
            self.validate_api_response(indices_data, "Market indices API")

            # Handle nested data structure
            idx_data = self.safe_get(indices_data, ['data', 'data', 'IDX_I'], {})
            
            id_to_field = {
                str(self.NIFTY_SCRIP_ID): {
                    'ltp': 'nifty_ltp',
                    'high': 'nifty_high',
                    'low': 'nifty_low'
                },
                str(self.VIX_SCRIP_ID): {'ltp': 'vix'},
                str(self.BANKNIFTY_SCRIP_ID): {'ltp': 'banknifty'}
            }
            
            for scrip_id, fields in id_to_field.items():
                if scrip_id in idx_data:
                    scrip_data = idx_data[scrip_id]
                    ohlc_data = scrip_data.get('ohlc', {})
                    
                    for api_field, output_field in fields.items():
                        if api_field == 'ltp':
                            value = scrip_data.get('last_price', 0)
                        elif api_field in ['high', 'low']:
                            value = ohlc_data.get(api_field, 0)
                        market_data[output_field] = self.safe_float(value)
                else:
                    for output_field in fields.values():
                        market_data[output_field] = 0.0

            return market_data

        except Exception as e:
            logger.error(f"Error in market data processing: {str(e)}")
            raise

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the dataset using TA-Lib"""
        try:
            if df.empty:
                return df
                
            historical_file = f'{self.UNPROCESSED_DATA_DIR}/historical_data.csv'
            if not os.path.exists(historical_file):
                return df
                
            historical_df = pd.read_csv(historical_file)
            historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
            
            result_df = df.copy()
            result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
            
            # Initialize technical indicator columns
            result_df['sma'] = 0.0
            result_df['macd'] = 0.0
            result_df['macd_signal'] = 0.0
            result_df['macd_histogram'] = 0.0
            result_df['option_rsi'] = 0.0
            result_df['nifty_rsi'] = 0.0
            result_df['atr'] = 0.0  # Initialize ATR column
            
            # Ensure expiry_date is present in both DataFrames
            if 'expiry_date' not in historical_df.columns:
                historical_df['expiry_date'] = None
            
            if 'expiry_date' not in result_df.columns:
                result_df['expiry_date'] = df['expiry_date']
            
            # Combine historical and new data with all necessary columns
            columns_to_keep = ['timestamp', 'strike_price', 'option_type', 'expiry_date', 
                             'nifty_ltp', 'nifty_high', 'nifty_low', 'option_ltp']
            
            combined_df = pd.concat([
                historical_df[columns_to_keep],
                result_df[columns_to_keep]
            ])
            
            combined_df = combined_df.drop_duplicates(
                subset=['timestamp', 'strike_price', 'option_type', 'expiry_date'],
                keep='last'
            )
            combined_df = combined_df.sort_values(['timestamp', 'strike_price', 'option_type'])
            
            # Calculate Nifty indicators
            nifty_hist = combined_df.drop_duplicates(subset=['timestamp'], keep='first')[
                ['timestamp', 'nifty_ltp', 'nifty_high', 'nifty_low']
            ]
            nifty_hist = nifty_hist[nifty_hist['nifty_ltp'] > 0]  # Filter out invalid data points
            nifty_hist = nifty_hist.sort_values('timestamp')
            
            nifty_prices = nifty_hist['nifty_ltp'].values
            nifty_high = nifty_hist['nifty_high'].values
            nifty_low = nifty_hist['nifty_low'].values
            
            # Calculate ATR with period 14 (standard)
            if len(nifty_prices) >= 14:
                atr = talib.ATR(nifty_high, nifty_low, nifty_prices, timeperiod=14)
                result_df['atr'] = round(atr[-1], 2)
            
            # Calculate SMA
            if len(nifty_prices) >= self.SMA_PERIOD:
                sma = talib.SMA(nifty_prices, timeperiod=self.SMA_PERIOD)
                result_df['sma'] = round(sma[-1], 2)
            
            # Calculate MACD with standard periods
            if len(nifty_prices) >= self.MACD_SLOW:  # Minimum required for slow period
                macd, signal, hist = talib.MACD(
                    nifty_prices,
                    fastperiod=self.MACD_FAST,    # Using class parameter
                    slowperiod=self.MACD_SLOW,    # Using class parameter
                    signalperiod=self.MACD_SIGNAL # Using class parameter
                )
                
                result_df['macd'] = round(macd[-1], 2)
                result_df['macd_signal'] = round(signal[-1], 2)
                result_df['macd_histogram'] = round(hist[-1], 2)
            
            # Calculate Nifty RSI
            if len(nifty_prices) >= self.RSI_PERIOD:
                nifty_rsi = talib.RSI(nifty_prices, timeperiod=self.RSI_PERIOD)
                result_df['nifty_rsi'] = round(nifty_rsi[-1], 2)
            
            # Calculate Option RSI for each unique combination
            unique_combinations = result_df[['strike_price', 'option_type', 'expiry_date']].drop_duplicates().values
            
            for (strike, opt_type, expiry) in unique_combinations:
                option_hist = combined_df[
                    (combined_df['strike_price'] == strike) & 
                    (combined_df['option_type'] == opt_type) &
                    (combined_df['expiry_date'] == expiry)
                ]
                option_hist = option_hist[option_hist['option_ltp'] > 0]  # Filter out invalid data points
                option_hist = option_hist.sort_values('timestamp')
                
                if option_hist.empty:
                    continue
                    
                hist_prices = option_hist['option_ltp'].values
                
                if len(hist_prices) >= self.RSI_PERIOD:
                    option_rsi = talib.RSI(hist_prices, timeperiod=self.RSI_PERIOD)
                    rsi_value = round(option_rsi[-1], 2)
                    
                    result_df.loc[
                        (result_df['strike_price'] == strike) & 
                        (result_df['option_type'] == opt_type) &
                        (result_df['expiry_date'] == expiry),
                        'option_rsi'
                    ] = rsi_value
            
            # Fill NaN values with 0
            result_df = result_df.fillna(0)
            return result_df
            
        except Exception as e:
            logger.error(f"Error in calculate_technical_indicators: {str(e)}")
            return df

    def manage_historical_data(self, new_data: Dict) -> pd.DataFrame:
        """Manage historical data retention and cleanup with optimized storage"""
        historical_file = f'{self.UNPROCESSED_DATA_DIR}/historical_data.csv'
        
        try:
            current_timestamp = new_data['timestamp']
            market_data = json.loads(new_data['market_data'])
            option_chain = json.loads(new_data['option_chain'])
            nifty_high = new_data['nifty_high']
            nifty_low = new_data['nifty_low']
            
            option_data = option_chain.get('data', {}).get('data', {})
            strikes_data = option_data.get('oc', {})
            
            last_price = float(option_data.get('last_price', 0))
            atm_strike = round(last_price / 50) * 50
            
            min_strike = atm_strike - (24 * 50)
            max_strike = atm_strike + (24 * 50)
            
            historical_records = []
            
            for strike_str, strike_data in strikes_data.items():
                strike = float(strike_str)
                if min_strike <= strike <= max_strike:
                    for side in ['ce', 'pe']:
                        if side in strike_data:
                            option = strike_data[side]
                            
                            record = {
                                'timestamp': current_timestamp,
                                'strike_price': strike,
                                'option_type': side.upper(),
                                'expiry_date': new_data['expiry_date'],
                                'nifty_ltp': market_data['nifty_ltp'],
                                'nifty_high': nifty_high,
                                'nifty_low': nifty_low,
                                'option_ltp': round(float(option.get('last_price', 0)), 2),
                                'oi': int(option.get('oi', 0))
                            }
                            historical_records.append(record)
            
            new_df = pd.DataFrame(historical_records)
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
            new_records_count = len(new_df)  # Count of new records being added
            
            historical_df = pd.DataFrame()
            if os.path.exists(historical_file):
                try:
                    historical_df = pd.read_csv(historical_file)
                    if set(historical_df.columns) == set(new_df.columns):
                        historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
                    else:
                        historical_df = pd.DataFrame()
                except Exception as e:
                    logger.error(f"Error reading historical data: {str(e)}")
                    historical_df = pd.DataFrame()
            
            historical_df = pd.concat([historical_df, new_df])
            
            cutoff_date = datetime.now() - timedelta(days=7)
            historical_df = historical_df[historical_df['timestamp'] > cutoff_date]
            
            historical_df = historical_df.drop_duplicates(
                subset=['timestamp', 'strike_price', 'option_type', 'expiry_date'],
                keep='last'
            )
            
            historical_df = historical_df.sort_values(['timestamp', 'strike_price', 'option_type'])
            
            historical_df.to_csv(historical_file, index=False)
            logger.info(f"Historical data updated [new records: {new_records_count}]")
            
            return historical_df
            
        except Exception as e:
            logger.error(f"Error managing historical data: {str(e)}")
            return pd.DataFrame()

    def send_telegram_message(self, message: str):
        """Send message to Telegram"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            logger.warning("Telegram configuration not found. Skipping notification.")
            return
            
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=data)
            response.raise_for_status()
            logger.info("Telegram notification sent successfully")
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {str(e)}")

    def get_intraday_data(self) -> Tuple[float, float]:
        """Fetch intraday minute data and return high/low values"""
        try:
            today = datetime.now()
            response = self.dhan.intraday_minute_data(
                security_id=self.NIFTY_SCRIP_ID,
                exchange_segment="IDX_I",
                instrument_type="INDEX",
                interval=5,
                from_date=today.strftime('%Y-%m-%d'),
                to_date=today.strftime('%Y-%m-%d')
            )
            
            if response.get('status') == 'success':
                data = response.get('data', {})
                if data and len(data.get('high', [])) >= 2:
                    # Get the second-to-last values for high and low
                    nifty_high = round(float(data['high'][-2]), 2)
                    nifty_low = round(float(data['low'][-2]), 2)
                    return nifty_high, nifty_low
                else:
                    raise Exception("Insufficient intraday data points")
            else:
                raise Exception(f"Failed to get intraday data: {response.get('remarks', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error fetching intraday data: {str(e)}")
            raise

    def validate_critical_data(self, data: Dict) -> bool:
        """Validate critical data fields and log errors if invalid"""
        critical_fields = {
            'nifty_ltp': data.get('nifty_ltp'),
            'nifty_high': data.get('nifty_high'),
            'nifty_low': data.get('nifty_low'),
            'india_vix': data.get('india_vix'),
            'banknifty': data.get('banknifty'),
            'strike_price': data.get('strike_price'),
            'option_ltp': data.get('option_ltp'),
            'ask_price': data.get('ask_price'),
            'bid_price': data.get('bid_price'),
            'pcr': data.get('pcr'),
            'oi': data.get('oi'),
            'volume': data.get('volume')
        }

        error_fields = []
        for field, value in critical_fields.items():
            try:
                if pd.isna(value) or value == 0 or not value:
                    error_fields.append(field)
            except:
                error_fields.append(field)

        if error_fields:
            error_msg = f"Invalid data detected at {self.current_timestamp}:\n"
            for field in error_fields:
                error_msg += f"- {field}: {critical_fields[field]}\n"
            
            # Only log the error, don't send notification here
            logger.error(error_msg)
            return False
        
        return True

    def check_data_freshness(self):
        """Check if data is being collected regularly"""
        try:
            if not os.path.exists(self.processed_file):
                return True  # First run of the day
            
            df = pd.read_csv(self.processed_file)
            if df.empty:
                return True
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            latest_timestamp = df['timestamp'].max()
            current_time = datetime.now()
            
            # Only check between 9:25 AM and 3:30 PM on trading days
            current_time_only = current_time.time()
            if (current_time_only < datetime.strptime('09:25:00', '%H:%M:%S').time() or 
                current_time_only > datetime.strptime('15:30:00', '%H:%M:%S').time()):
                return True
            
            time_diff = (current_time - latest_timestamp).total_seconds() / 60
            if time_diff > 8:
                error_msg = (f"Data collection gap detected!\n"
                           f"Last entry: {latest_timestamp}\n"
                           f"Current time: {current_time}\n"
                           f"Gap: {int(time_diff)} minutes")
                logger.error(error_msg)
                self.send_telegram_message(f"🚨 <b>Data Collection Gap</b>\n\n{error_msg}")
            
            # Always return True to continue processing
            return True
            
        except Exception as e:
            logger.error(f"Error checking data freshness: {str(e)}")
            # Return True even on error to continue processing
            return True

    def get_next_trading_day(self, start_date):
        """Get the next trading day from the given date"""
        next_day = start_date + timedelta(days=1)
        while next_day.weekday() >= 5 or next_day in self.TRADING_HOLIDAYS_2025:
            next_day += timedelta(days=1)
        return next_day

    def get_last_trading_day(self, target_date):
        """Get the last trading day before the given date"""
        check_date = target_date - timedelta(days=1)
        while check_date.weekday() >= 5 or check_date in self.TRADING_HOLIDAYS_2025:
            check_date -= timedelta(days=1)
        return check_date

    def should_fetch_next_expiry(self, current_expiry):
        """
        Determine if we should fetch next expiry data based on:
        1. If today is expiry day
        2. If tomorrow is a holiday and day after is expiry
        3. If next trading day is expiry
        4. If expiry falls after weekend/holidays
        5. If today is the last trading day before expiry
        """
        try:
            # Convert expiry string to date object
            expiry_date = datetime.strptime(current_expiry, '%Y-%m-%d').date()
            
            # If today is expiry day, we should use next expiry data
            if self.today == expiry_date:
                logger.info(f"Today {self.today} is expiry day")
                return True
            
            # Get next trading day after today
            next_trading_day = self.get_next_trading_day(self.today)
            
            # If next trading day is expiry, we should fetch next expiry data
            if next_trading_day == expiry_date:
                logger.info(f"Next trading day {next_trading_day} is expiry day")
                return True
            
            # Check if expiry is after a non-trading period (weekend or holidays)
            last_trading_day_before_expiry = self.get_last_trading_day(expiry_date)
            
            # If today is the last trading day before expiry
            if self.today == last_trading_day_before_expiry:
                logger.info(f"Today {self.today} is the last trading day before expiry {expiry_date}")
                return True
            
            # Additional check for consecutive holidays
            next_day = self.today + timedelta(days=1)
            days_until_expiry = (expiry_date - self.today).days
            
            # If we're within 5 days of expiry and next day is not a trading day
            if days_until_expiry <= 5 and (next_day.weekday() >= 5 or next_day in self.TRADING_HOLIDAYS_2025):
                next_trading = self.get_next_trading_day(self.today)
                if next_trading == expiry_date:
                    logger.info(f"Today {self.today} is the last trading day before non-trading period leading to expiry {expiry_date}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in should_fetch_next_expiry: {str(e)}")
            return False

    def collect_and_process_data(self):
        """Main function to collect and process option chain data"""
        retry_count = 0
        last_error = None
        
        while retry_count < self.MAX_RETRIES:
            try:
                # Check data freshness at the start
                self.check_data_freshness()
                
                if retry_count > 0:
                    logger.warning(f"Retry attempt {retry_count}/{self.MAX_RETRIES}")
                    self.send_telegram_message(f"🔄 <b>Retrying Data Collection</b>\n\nAttempt {retry_count}/{self.MAX_RETRIES}")
                    time.sleep(self.RETRY_DELAY)
                
                logger.info("Starting data collection cycle")
                
                # Reset current time for this collection cycle
                self._current_time = None
                
                # Check token expiry before each data collection
                self.check_token_expiry()
                
                # Get current expiry date
                current_expiry = self.get_current_expiry()
                logger.info(f"Processing expiry: {current_expiry}")

                # Check if we should fetch next expiry data
                should_fetch_next = self.should_fetch_next_expiry(current_expiry)
                
                # If today is expiry day, use next expiry as primary data
                if self.today == datetime.strptime(current_expiry, '%Y-%m-%d').date():
                    current_expiry = self.get_next_expiry()
                    logger.info(f"Today is expiry day, switched to next expiry: {current_expiry}")
                    should_fetch_next = False  # No need to fetch next expiry again
                elif should_fetch_next:
                    logger.info("Will fetch next expiry data (holidays/weekend ahead)")
                
                # Process current expiry
                market_data = self.get_market_data()
                if not market_data:
                    raise Exception("Failed to fetch market data")

                # Fetch intraday data once
                nifty_high, nifty_low = self.get_intraday_data()

                option_chain_data = self.get_option_chain(current_expiry)
                if not option_chain_data:
                    raise Exception("Failed to fetch option chain data")

                # First process the option chain with current data
                processed_data = self.process_option_chain(market_data, option_chain_data, current_expiry, nifty_high, nifty_low)
                if not processed_data:
                    raise Exception("No processed data available")

                # Then update historical data
                historical_data = {
                    'timestamp': self.current_timestamp,
                    'market_data': json.dumps(market_data),
                    'option_chain': json.dumps(option_chain_data),
                    'expiry_date': current_expiry,
                    'nifty_high': nifty_high,
                    'nifty_low': nifty_low
                }
                
                historical_df = self.manage_historical_data(historical_data)
                logger.info(f"Data collection cycle completed [records: {len(processed_data)}]")

                # If we should fetch next expiry data
                if should_fetch_next:
                    logger.debug("Fetching next expiry data...")
                    time.sleep(5)
                    
                    # Get next expiry date
                    next_expiry = self.get_next_expiry()
                    logger.info(f"Processing next expiry: {next_expiry}")
                    
                    # Fetch option chain for next expiry
                    next_option_chain_data = self.get_option_chain(next_expiry)
                    if not next_option_chain_data:
                        logger.warning("Failed to fetch next expiry option chain data")
                        self.send_telegram_message("⚠️ <b>Warning</b>\n\nFailed to fetch next expiry option chain data")
                    else:
                        # Process next expiry data
                        next_processed_data = self.process_option_chain(market_data, next_option_chain_data, next_expiry, nifty_high, nifty_low)
                        if next_processed_data:
                            # Update historical data with next expiry data
                            next_historical_data = {
                                'timestamp': self.current_timestamp,
                                'market_data': json.dumps(market_data),
                                'option_chain': json.dumps(next_option_chain_data),
                                'expiry_date': next_expiry,
                                'nifty_high': nifty_high,
                                'nifty_low': nifty_low
                            }
                            historical_df = self.manage_historical_data(next_historical_data)
                            logger.info(f"Next expiry data processed [records: {len(next_processed_data)}]")
                        else:
                            logger.warning("Failed to process next expiry data")
                            self.send_telegram_message("⚠️ <b>Warning</b>\n\nFailed to process next expiry data")

                df = pd.DataFrame(processed_data)
                df = df.sort_values('distance')
                
                df_with_indicators = self.calculate_technical_indicators(df)
                
                if os.path.exists(self.processed_file):
                    try:
                        existing_df = pd.read_csv(self.processed_file)
                        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                        df_with_indicators['timestamp'] = pd.to_datetime(df_with_indicators['timestamp'])
                        
                        if not existing_df.empty and not df_with_indicators.empty:
                            # Ensure all required columns exist
                            existing_df = self.ensure_dataframe_columns(existing_df, self.REQUIRED_COLUMNS)
                            df_with_indicators = self.ensure_dataframe_columns(df_with_indicators, self.REQUIRED_COLUMNS)
                            
                            # Ensure expiry_date is present
                            if 'expiry_date' not in existing_df.columns:
                                existing_df['expiry_date'] = current_expiry
                            if 'expiry_date' not in df_with_indicators.columns:
                                df_with_indicators['expiry_date'] = current_expiry
                            
                            latest_existing_time = existing_df['timestamp'].max()
                            df_with_indicators = df_with_indicators[df_with_indicators['timestamp'] > latest_existing_time]
                            
                            df_with_indicators = pd.concat([existing_df, df_with_indicators], sort=False)
                            
                            df_with_indicators = df_with_indicators.drop_duplicates(
                                subset=['timestamp', 'strike_price', 'option_type', 'expiry_date'],
                                keep='last'
                            )
                    except Exception as e:
                        error_msg = f"Error processing existing data: {str(e)}"
                        logger.error(error_msg)
                        self.send_telegram_message(f"❌ <b>Data Processing Error</b>\n\n{error_msg}")
                        raise
                
                df_with_indicators = df_with_indicators.sort_values(['timestamp', 'distance'], ascending=[True, True])
                
                df_with_indicators.to_csv(self.processed_file, index=False)
                logger.info(f"Processed data saved to {self.processed_file} with {len(df_with_indicators)} records")
                
                # If we reach here, everything was successful
                if retry_count > 0:
                    success_msg = f"✅ <b>Data Collection Succeeded</b>\n\nSuccessful after {retry_count} retries"
                    logger.info(f"Data collection succeeded after {retry_count} retries")
                    self.send_telegram_message(success_msg)
                
                return True
                
            except Exception as e:
                retry_count += 1
                last_error = str(e)
                logger.error(f"Error in data collection (attempt {retry_count}): {last_error}")
                
                if retry_count >= self.MAX_RETRIES:
                    error_msg = (
                        f"❌ <b>Data Collection Failed</b>\n\n"
                        f"Failed after {retry_count} attempts.\n"
                        f"Last error: {last_error}"
                    )
                    logger.error(f"Data collection failed after {retry_count} attempts")
                    self.send_telegram_message(error_msg)
                    return False

    def get_option_chain(self, expiry_date: str) -> Dict:
        """Fetch option chain data"""
        try:
            response = self.dhan.option_chain(
                self.NIFTY_SCRIP_ID,
                "IDX_I",
                expiry_date
            )
            
            self.validate_api_response(response, "Option chain API")
            return response
            
        except Exception as e:
            logger.error(f"Failed to fetch option chain: {str(e)}")
            self.send_telegram_message(f"❌ <b>Option Chain Error</b>\n\n{str(e)}")
            return None

    def process_option_chain(self, market_data: Dict, option_chain_data: Dict, expiry_date: str, nifty_high: float, nifty_low: float) -> List[Dict]:
        """Process option chain data and return list of processed records"""
        try:
            # Use safe_get for nested dictionary access
            option_chain = self.safe_get(option_chain_data, ['data'], {})
            if not option_chain:
                raise Exception("Empty option chain data received")
            
            option_data = self.safe_get(option_chain, ['data'], {})
            if not option_data:
                raise Exception("No data field in option chain")
            
            last_price = self.safe_float(option_data.get('last_price'))
            if last_price == 0:
                raise Exception("Invalid last price from option chain")
            
            atm_strike = round(last_price / 50) * 50
            min_strike = atm_strike - (self.NUM_STRIKES * 50)
            max_strike = atm_strike + (self.NUM_STRIKES * 50)
            
            strikes_data = option_data.get('oc', {})
            if not strikes_data:
                raise Exception("No options data found in option chain")
            
            # Calculate total OI for PCR
            total_pe_oi = total_ce_oi = 0
            for strike_data in strikes_data.values():
                total_pe_oi += self.safe_int(self.safe_get(strike_data, ['pe', 'oi']))
                total_ce_oi += self.safe_int(self.safe_get(strike_data, ['ce', 'oi']))
            
            pcr_oi = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0
            
            # Get previous OI data BEFORE updating historical data
            previous_oi = {}
            has_previous_data = False
            
            if os.path.exists(self.historical_file):
                try:
                    # Read only last 5000 rows for efficiency
                    file_size = sum(1 for _ in open(self.historical_file))
                    skip_rows = max(0, file_size - self.HISTORICAL_DATA_ROWS)
                    
                    historical_df = pd.read_csv(
                        self.historical_file,
                        skiprows=lambda x: x > 0 and x <= skip_rows
                    )
                    historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
                    
                    if not historical_df.empty:
                        # Get data only for the current expiry date
                        expiry_data = historical_df[historical_df['expiry_date'] == expiry_date]
                        
                        if not expiry_data.empty:
                            has_previous_data = True
                            # Get the latest timestamp for this expiry
                            latest_timestamp = expiry_data['timestamp'].max()
                            
                            # Get data for the latest timestamp and current expiry
                            latest_data = expiry_data[expiry_data['timestamp'] == latest_timestamp]
                            
                            # Store previous OI values with expiry consideration
                            for _, row in latest_data.iterrows():
                                strike = self.safe_float(row['strike_price'])
                                option_type = row['option_type']
                                oi = self.safe_int(row['oi'])
                                previous_oi[(strike, option_type, expiry_date)] = oi
                            
                except Exception as e:
                    logger.error(f"Error reading previous OI data: {str(e)}")
                    has_previous_data = False
                    previous_oi = {}
            
            processed_data = []
            validation_errors = set()  # Use set to store unique error messages
            
            for strike_str, strike_data in strikes_data.items():
                try:
                    # Add type checking for strike_str
                    if not isinstance(strike_str, (str, int, float)):
                        logger.error(f"Invalid strike price type: {type(strike_str)}")
                        continue
                        
                    try:
                        strike = self.safe_float(strike_str)
                    except ValueError:
                        logger.error(f"Could not convert strike price to float: {strike_str}")
                        continue
                    
                    if min_strike <= strike <= max_strike:
                        for side in ['ce', 'pe']:
                            if side in strike_data:
                                option_data = strike_data[side]
                                greeks = option_data.get('greeks', {})
                                distance = (atm_strike - strike) if side == 'ce' else (strike - atm_strike)
                                
                                # Get current OI and calculate change
                                current_oi = self.safe_int(option_data.get('oi'))
                                option_type = side.upper()
                                
                                # Calculate OI change
                                oi_change = 0
                                if has_previous_data:
                                    prev_oi = previous_oi.get((strike, option_type, expiry_date), 0)
                                    oi_change = current_oi - prev_oi
                                
                                # Process common option data
                                last_price, ask_price, bid_price, volume = self.process_option_data(option_data)
                                
                                row_data = {
                                    'timestamp': self.current_timestamp,
                                    'nifty_ltp': self.safe_float(market_data['nifty_ltp']),
                                    'nifty_high': self.safe_float(nifty_high),
                                    'nifty_low': self.safe_float(nifty_low),
                                    'india_vix': self.safe_float(market_data['vix']),
                                    'banknifty': self.safe_float(market_data['banknifty']),
                                    'expiry_date': expiry_date,
                                    'strike_price': strike,
                                    'distance': self.safe_float(distance),
                                    'option_type': option_type,
                                    'option_ltp': last_price,
                                    'ask_price': ask_price,
                                    'bid_price': bid_price,
                                    'ask_bid_spread': self.safe_float(ask_price - bid_price),
                                    'pcr': pcr_oi,
                                    'days_to_expiry': (datetime.strptime(expiry_date, '%Y-%m-%d').date() - self.today).days,
                                    'delta': self.safe_float(greeks.get('delta'), precision=self.GREEK_PRECISION),
                                    'theta': self.safe_float(greeks.get('theta'), precision=self.GREEK_PRECISION),
                                    'gamma': self.safe_float(greeks.get('gamma'), precision=self.GREEK_PRECISION),
                                    'vega': self.safe_float(greeks.get('vega'), precision=self.GREEK_PRECISION),
                                    'iv': self.safe_float(option_data.get('implied_volatility'), precision=self.GREEK_PRECISION),
                                    'oi': current_oi,
                                    'oi_change': oi_change,
                                    'volume': volume
                                }
                                
                                # Validate critical data and collect errors
                                if not self.validate_critical_data(row_data):
                                    error_msg = f"Strike {strike} {option_type} has invalid critical data"
                                    validation_errors.add(error_msg)
                                
                                processed_data.append(row_data)
                                
                except Exception as e:
                    logger.error(f"Error processing strike {strike_str}: {str(e)}")
                    continue
            
            if not processed_data:
                raise Exception("Invalid options data processed")
                
            # Send a single consolidated notification for all validation errors
            if validation_errors:
                error_msg = (f"⚠️ <b>Data Validation Errors at {self.current_timestamp}</b>\n\n"
                            f"{len(validation_errors)} validation issues found:\n\n"
                            f"{'-' * 40}\n")
                error_msg += "\n".join(list(validation_errors)[:10])  # Limit to first 10 errors
                if len(validation_errors) > 10:
                    error_msg += f"\n\n... and {len(validation_errors) - 10} more errors"
                self.send_telegram_message(error_msg)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in process_option_chain: {str(e)}")
            return None

    def run_scheduler(self):
        """Run the data collection once"""
        try:
            logger.info("Initializing data collection")
            success = self.collect_and_process_data()
            if success:
                logger.info("Data collection completed successfully")
            
        except Exception as e:
            error_message = f"Fatal error in execution: {str(e)}"
            logger.error(error_message)
            self.send_telegram_message(f"❌ <b>Option Chain Collector Fatal Error</b>\n\n{error_message}")

if __name__ == "__main__":
    collector = NiftyOptionChainCollector()
    collector.run_scheduler() 
