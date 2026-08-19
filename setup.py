from setuptools import setup, find_packages

setup(
    name="ai_trading_bot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "stable-baselines3",
        "pandas",
        "numpy",
        "ta-lib",
        "python-dotenv",
        "requests",
        "matplotlib",
        "tqdm",
    ],
    author="KushalAzza",
    description="Reinforcement learning-based options trading system for algorithmic trading on Indian markets",
    python_requires=">=3.8",
) 