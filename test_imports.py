import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from scripts.data_collection import NiftyOptionChainCollector

if __name__ == "__main__":
    collector = NiftyOptionChainCollector()
    print("Imports successful!") 