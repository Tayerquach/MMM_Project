import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# Function to plot forecast series
def plot_forecast(train_df, test_df, prediction: np.array, target_name, title, xlabel, ylabel):
    """
    This function plots the train, test, and forecast values with date range selection.
    """
    # Combine train and test to get full date range
    full_df = pd.concat([train_df, test_df])
    
    # Get min and max dates
    min_date = full_df.index.min()
    max_date = full_df.index.max()

    # Create figure
    fig = go.Figure()

    # Add training data (Blue)
    fig.add_trace(go.Scatter(
        x=train_df.index, 
        y=train_df[target_name], 
        mode='lines', 
        name='Train',
        line=dict(color='blue')
    ))

    # Add test data (Red)
    fig.add_trace(go.Scatter(
        x=test_df.index, 
        y=test_df[target_name], 
        mode='lines', 
        name='Test',
        line=dict(color='red')
    ))

    # Add forecasted values (Green)
    fig.add_trace(go.Scatter(
        x=test_df.index,  
        y=prediction, 
        mode='lines', 
        name='Forecast',
        line=dict(color='green', dash='dot')
    ))

    # Add dropdown menus for filtering
    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {"label": "Show Test Data", "method": "update", "args": [{"visible": [True, True, True]}]},
                    {"label": "Hide Test Data", "method": "update", "args": [{"visible": [True, False, True]}]}
                ],
                "direction": "down",
                "showactive": True,
                "x": 1.0, 
                "y": 1.15, 
                "xanchor": "right",
                "yanchor": "top"
            }
        ],
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        legend=dict(x=0.02, y=0.98),
        template="plotly_white",
        
        # Add range slider for selecting date range
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    {"count": 7, "label": "1W", "step": "day", "stepmode": "backward"},
                    {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
                    {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
                    {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                    {"step": "all", "label": "All"}
                ]
            ),
            rangeslider=dict(visible=True),  # Enable interactive date filtering
            type="date"
        )
    )

    return fig  # Return the figure instead of showing it

def format_thousands_millions(x, pos) -> str:
    if abs(x) >= 1e6:
        return "{:.1f}M".format(x * 1e-6)
    elif abs(x) >= 1e3:
        return "{:.1f}K".format(x * 1e-3)
    else:
        return "{:.1f}".format(x)

def plot_spend_response_curve(
    channel, spend_response_df, response_name, 
    average_spend=0, average_response=0, 
    max_spend=0, max_response=0, 
    optimized_spend=None, optimized_response=None, 
    figure_size=(15, 6)
):
    statistics_spend_response_df = pd.DataFrame({
        'average_spend': [average_spend], 
        'average_response': [average_response],
        'max_spend': [max_spend],
        'max_response': [max_response],
    })

    fig, ax = plt.subplots(figsize=figure_size)
    sns.lineplot(data=spend_response_df, x='spend', y='response')
    plt.xlabel('Spend (with adstock effect)', fontsize=20)
    plt.ylabel(response_name, fontsize=20)

    # Scatter plots
    sns.scatterplot(data=statistics_spend_response_df, x='average_spend', y='average_response', color='blue', s=100, label="Average")
    sns.scatterplot(data=statistics_spend_response_df, x='max_spend', y='max_response', color='red', s=100, label="Max")

    # Vertical and horizontal reference lines
    avg_vline = plt.axvline(x=statistics_spend_response_df['average_spend'][0], linestyle='solid', color='blue', alpha=0.7, label="Average Spend")
    max_vline = plt.axvline(x=statistics_spend_response_df['max_spend'][0], linestyle='dotted', color='red', alpha=0.7, label="Max Spend")

    # Format x-axis and y-axis
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(format_thousands_millions))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(format_thousands_millions))

    if optimized_spend is not None and optimized_response is not None:
        optimized_spend_response_df = pd.DataFrame({
            'optimized_spend': [optimized_spend], 
            'optimized_response': [optimized_response]
        })
        sns.scatterplot(data=optimized_spend_response_df, x='optimized_spend', y='optimized_response', color='green', s=150, label="Optimal")
        optimal_vline = plt.axvline(x=optimized_spend_response_df['optimized_spend'][0], linestyle='dotted', alpha=0.5, color='green', label="Optimal Spend")
        plt.axhline(y=optimized_spend_response_df['optimized_response'][0], linestyle='dotted', alpha=0.5, color='green')
        plt.legend(handles=[avg_vline, max_vline, optimal_vline])
    else:
        plt.legend(handles=[avg_vline, max_vline])

    plt.title(f"{channel}: Response Curve", fontsize=25)
    plt.xticks(fontsize=18)  # Increase x-axis tick font size
    plt.yticks(fontsize=18)  # Increase y-axis tick font size

    return fig 


def plot_response_decomposition(contribution_df):
    """
    Creates a horizontal bar chart (Waterfall) showing response decomposition by channels.

    Args:
        contribution_df (pd.DataFrame): DataFrame with 'channels' and 'effect_share' columns.

    Returns:
        Matplotlib figure for Streamlit.
    """
    # Sort data
    contribution_df = contribution_df.sort_values(by="effect_share", ascending=True)  

    # Convert effect_share to percentage
    contribution_df['effect_share'] = round(contribution_df['effect_share'] * 100, 2)

    # Compute cumulative positions for stacking
    cumulative = np.zeros(len(contribution_df))
    cumulative[1:] = np.cumsum(contribution_df['effect_share'][:-1])  # Shifted cumulative sum

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot bars
    bars = ax.barh(contribution_df['channels'], contribution_df['effect_share'], left=cumulative, color='#82e0df')

    # Add text labels inside bars (Only Show Total Effect Percentage)
    for bar, percent in zip(bars, contribution_df['effect_share']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, 
                f"{percent:.1f}%", ha='center', va='center', fontsize=18, color='black')

    # Remove unnecessary axis lines (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)  # Removes y-axis line

    # Label axes
    ax.set_xlabel("Cumulative Contribution", fontsize=18)
    ax.set_ylabel("Channel", fontsize=18)
    plt.title("Response Decomposition Waterfall by Channels", fontsize=25)
    plt.xticks(fontsize=18)  # Increase x-axis tick font size
    plt.yticks(fontsize=18)  # Increase y-axis tick font size


    return fig 





