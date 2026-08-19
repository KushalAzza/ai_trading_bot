import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import sys
from pathlib import Path

# Add the project root directory to Python path for imports
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.utils import (
    setup_logging,
    ensure_directory_exists,
    PROCESSED_DATA_DIR,
    LOGS_DIR
)

class DataArchiver:
    def __init__(self, retention_days: int = 30):
        """Initialize DataArchiver with specified retention period.
        
        Args:
            retention_days (int): Number of days to retain data before archiving
        """
        self.logger = setup_logging('data_archiver')
        self.retention_days = retention_days
        
        # Use constants from utils.py and derive paths
        self.ARCHIVE_DIR = f'{PROCESSED_DATA_DIR}/archive'
        self.TRAINING_DATA_FILE = f'{PROCESSED_DATA_DIR}/training_data.csv'
        self.PROCESSED_DATA_FILE = f'{PROCESSED_DATA_DIR}/processed_data.csv'
        
        # Create directories using utility function
        ensure_directory_exists(self.ARCHIVE_DIR)
        self.logger.info(f"Initialized DataArchiver with {retention_days} days retention")

    def deduplicate_and_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate and sort DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to process
            
        Returns:
            pd.DataFrame: Deduplicated and sorted DataFrame
        """
        try:
            initial_len = len(df)
            deduped_df = df.drop_duplicates(
                subset=['timestamp', 'strike_price', 'option_type', 'expiry_date'],
                keep='last'
            ).sort_values('timestamp')
            
            removed_count = initial_len - len(deduped_df)
            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} duplicate records")
            
            return deduped_df
            
        except Exception as e:
            self.logger.error(f"Error in deduplication: {str(e)}")
            raise

    def update_training_data(self):
        """Move data older than 1 day from processed_data.csv to training_data.csv"""
        try:
            if not os.path.exists(self.PROCESSED_DATA_FILE):
                self.logger.error("Processed data file not found")
                return

            # Read current processed data
            df = pd.read_csv(self.PROCESSED_DATA_FILE)
            if df.empty:
                self.logger.info("No data to process")
                return

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff_date = datetime.now() - timedelta(days=1)
            current_data = df[df['timestamp'] >= cutoff_date]
            training_data = df[df['timestamp'] < cutoff_date]

            # Update files
            if not current_data.empty:
                current_data.to_csv(self.PROCESSED_DATA_FILE, index=False)
                self.logger.info(f"Retained {len(current_data)} records in processed_data.csv")

            if not training_data.empty:
                if os.path.exists(self.TRAINING_DATA_FILE):
                    existing_training = pd.read_csv(self.TRAINING_DATA_FILE)
                    existing_training['timestamp'] = pd.to_datetime(existing_training['timestamp'])
                    training_data = pd.concat([existing_training, training_data])

                training_data = self.deduplicate_and_sort(training_data)
                training_data.to_csv(self.TRAINING_DATA_FILE, index=False)
                self.logger.info(f"Added {len(training_data)} records to training_data.csv")

        except Exception as e:
            self.logger.error(f"Error updating training data: {str(e)}")
            raise

    def archive_training_data(self):
        """Archive data older than RETENTION_DAYS from training_data.csv to monthly files"""
        try:
            if not os.path.exists(self.TRAINING_DATA_FILE):
                self.logger.error("Training data file not found")
                return

            # Read training data
            df = pd.read_csv(self.TRAINING_DATA_FILE)
            if df.empty:
                self.logger.info("No training data to archive")
                return

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            current_data = df[df['timestamp'] >= cutoff_date]
            archive_data = df[df['timestamp'] < cutoff_date]
            
            if archive_data.empty:
                self.logger.info("No training data to archive")
                return
            
            # Archive by month
            for (year, month), month_data in archive_data.groupby([
                archive_data['timestamp'].dt.year,
                archive_data['timestamp'].dt.month
            ]):
                archive_file = f'{self.ARCHIVE_DIR}/training_data_{year}{month:02d}.csv'
                
                if os.path.exists(archive_file):
                    existing_archive = pd.read_csv(archive_file)
                    existing_archive['timestamp'] = pd.to_datetime(existing_archive['timestamp'])
                    month_data = pd.concat([existing_archive, month_data])
                
                month_data = self.deduplicate_and_sort(month_data)
                month_data.to_csv(archive_file, index=False)
                self.logger.info(f"Archived {len(month_data)} records to {archive_file}")
            
            # Update training data file with remaining data
            if not current_data.empty:
                current_data = self.deduplicate_and_sort(current_data)
                current_data.to_csv(self.TRAINING_DATA_FILE, index=False)
                self.logger.info(f"Retained {len(current_data)} records in training_data.csv")
            
        except Exception as e:
            self.logger.error(f"Error during archiving training data: {str(e)}")
            raise

    def run(self):
        """Run both data management operations"""
        try:
            self.logger.info("Starting data management process...")
            self.update_training_data()
            self.archive_training_data()
            self.logger.info("Data management process completed successfully")
        except Exception as e:
            self.logger.error(f"Error in data management process: {str(e)}")
            raise

if __name__ == "__main__":
    archiver = DataArchiver(retention_days=30)
    archiver.run()