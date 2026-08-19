import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from scripts.utils import MODEL_DIR
import numpy as np
import pandas as pd

def load_training_metadata():
    """Load training metadata from the JSON file."""
    metadata_path = os.path.join(MODEL_DIR, 'training_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None

def create_training_visualization():
    """Create comprehensive interactive visualization of training metrics."""
    data = load_training_metadata()
    if not data:
        print("No training data found.")
        return

    # Create subplots: 4 plots and 2 tables
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Training Rewards',
            'Learning Curves',
            'Policy and Value Losses',
            'Explained Variance',
            'Training Information',
            'Hyperparameters'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "table"}, {"type": "table"}]
        ],
        vertical_spacing=0.15
    )

    # Extract metrics
    timestamps = data.get('timestamps', [])
    rewards = data.get('rewards', [])
    value_losses = data.get('value_losses', [])
    policy_losses = data.get('policy_losses', [])
    explained_variances = data.get('explained_variances', [])
    metadata = data.get('metadata', {})

    # 1. Episode Rewards
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=rewards,
            name='Episode Rewards',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    # 2. Learning Curves (Rolling mean of rewards)
    window = min(50, len(rewards))
    if window > 0:
        rolling_mean = pd.Series(rewards).rolling(window=window).mean()
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=rolling_mean,
                name=f'Rolling Mean (window={window})',
                line=dict(color='green')
            ),
            row=1, col=2
        )

    # 3. Policy and Value Losses
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=policy_losses,
            name='Policy Loss',
            line=dict(color='red')
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=value_losses,
            name='Value Loss',
            line=dict(color='orange')
        ),
        row=2, col=1
    )

    # 4. Explained Variance
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=explained_variances,
            name='Explained Variance',
            line=dict(color='purple')
        ),
        row=2, col=2
    )

    # 5. Training Information Table
    training_info = [
        ['Total Timesteps', metadata.get('total_timesteps_trained', 'N/A')],
        ['Training Duration', f"{timestamps[-1]} seconds"],
        ['Last Training Time', metadata.get('last_training_time', 'N/A')],
        ['Is Retraining', str(metadata.get('is_retraining', 'N/A'))],
        ['Best Reward', f"{max(rewards):.4f}"],
        ['Final Reward', f"{rewards[-1]:.4f}"],
        ['Final Explained Variance', f"{explained_variances[-1]:.4f}"]
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Metric', 'Value'],
                font=dict(size=12),
                align='left'
            ),
            cells=dict(
                values=list(zip(*training_info)),
                font=dict(size=11),
                align='left'
            )
        ),
        row=3, col=1
    )

    # 6. Hyperparameters Table
    hyperparams = metadata.get('hyperparameters', {})
    if hyperparams:
        param_info = [[k, str(v)] for k, v in hyperparams.items()]
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Parameter', 'Value'],
                    font=dict(size=12),
                    align='left'
                ),
                cells=dict(
                    values=list(zip(*param_info)),
                    font=dict(size=11),
                    align='left'
                )
            ),
            row=3, col=2
        )

    # Update layout
    fig.update_layout(
        height=1200,
        title_text='Training Metrics and Information',
        showlegend=True,
        template='plotly_white'
    )

    # Update axes labels
    fig.update_xaxes(title_text='Time (seconds)', row=1, col=1)
    fig.update_xaxes(title_text='Time (seconds)', row=1, col=2)
    fig.update_xaxes(title_text='Time (seconds)', row=2, col=1)
    fig.update_xaxes(title_text='Time (seconds)', row=2, col=2)
    
    fig.update_yaxes(title_text='Reward', row=1, col=1)
    fig.update_yaxes(title_text='Rolling Mean Reward', row=1, col=2)
    fig.update_yaxes(title_text='Loss', row=2, col=1)
    fig.update_yaxes(title_text='Explained Variance', row=2, col=2)

    # Save the visualization
    output_path = os.path.join(MODEL_DIR, 'training_visualization.html')
    fig.write_html(output_path)
    print(f"\nTraining visualization saved to {output_path}")

    # Print summary to console
    print("\nTraining Summary:")
    print(f"Total training time: {timestamps[-1]} seconds")
    print(f"Final reward: {rewards[-1]:.2f}")
    print(f"Best reward: {max(rewards):.2f}")
    print(f"Final explained variance: {explained_variances[-1]:.3f}")
    
    print("\nHyperparameters:")
    for param, value in hyperparams.items():
        print(f"{param}: {value}")

if __name__ == '__main__':
    create_training_visualization() 