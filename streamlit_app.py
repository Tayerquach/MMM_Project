import pickle
import streamlit as st
import pandas as pd
import numpy as np
from utils.metrics import mape_metrics, rmse_metrics
from utils.plot_helpers import plot_forecast, plot_response_decomposition, plot_spend_response_curve  # Replace 'your_script' with the actual script filename
# Load and display the logo next to "Marketing Science"
from PIL import Image
from utils.config import MEDIA_CHANNELS
import plotly.express as px

# Expand Streamlit working area width
st.markdown(
    """
    <style>
        .main .block-container {
            max-width: 100%;  /* Adjust width (default is ~70%) */
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


###### Sidebar menu

# Load the logo (Update the filename if needed)
logo_path = "photo/marsci_logo.png"  # Ensure the correct filename
logo = Image.open(logo_path)

# Sidebar with aligned logo + bigger text
with st.sidebar:
    col1, col2 = st.columns([1, 1])  # Adjust ratio for spacing
    with col1:
        st.image(logo, width=150)  # Adjust width as needed
    with col2:
        st.markdown(
            "<h1 style='font-size: 28px; margin-top: 10px;'>MarSci</h1>",
            unsafe_allow_html=True
        )

menu_options = ["Home", "Conversion Prediction", "Response Curve", "Budget Optimization"]
selected_menu = st.sidebar.selectbox("Select a Section", menu_options)


# Home Page - Load and Display Markdown Content
if selected_menu == "Home":
    # Load markdown content from introduction.md
    with open("introduction.md", "r", encoding="utf-8") as file:
        intro_content = file.read()

    # Display markdown content in Streamlit
    st.markdown(intro_content, unsafe_allow_html=True)

target_name = 'conversion'
# Mapping user-friendly names to actual model identifiers
model_mapping = {
    "Ridge Regression": "ridge",
    "LSTM": "lstm",
    "Facebook Prophet": "prophet",
    "SARIMAX": "sarimax"
}



# Load Data from CSV
# @st.cache_data
def load_data():
    train_df = pd.read_csv("data/train_data.csv", index_col=0, parse_dates=True)
    test_df = pd.read_csv("data/test_data.csv", index_col=0, parse_dates=True)
    return train_df, test_df


# Display content based on the selected menu option
if selected_menu == "Conversion Prediction":
    st.title("📈 Conversion Prediction")

    # Dropdown for model selection (shows user-friendly names)
    selected_model_name = st.selectbox("Select Model:", list(model_mapping.keys()), index=0)

    # Get the corresponding internal model identifier
    model_name = model_mapping[selected_model_name] 

    # Load Predictions from NumPy file
    # @st.cache_data
    def load_prediction():
        return np.load(f"data/{model_name}_predictions.npy")  # Load NumPy file

    # Load data and prediction
    train_df, test_df = load_data()
    predictions = load_prediction()  # Load saved predictions
     

    # Get actual values from the test dataset
    actual_values = test_df[target_name].values

    rmse = rmse_metrics(actual_values, predictions)
    mape = mape_metrics(actual_values, predictions)

    # Display Metrics in Streamlit
    st.header(f"Performance Metrics for ({selected_model_name})")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}")

    with col2:
        st.metric(label="Mean Absolute Percentage Error (MAPE)", value=f"{mape:.2f}%")


    # Generate the forecast plot
    title = 'Daily User Forecast'
    xlabel = 'Date'
    ylabel = 'Number of Users'
    fig = plot_forecast(train_df, test_df, predictions, target_name, title, xlabel, ylabel)

    # Display Plotly figure
    st.plotly_chart(fig, use_container_width=True)

elif selected_menu == "Response Curve":
    st.title("📊 Response Curve Analysis")

    with open("data/spend_response_curve_dict.pkl", "rb") as f:
        parameters = pickle.load(f)

    # User Inputs for Spend & Response Statistics
    channel = st.selectbox("Select Media Channel:", MEDIA_CHANNELS)
    response_name = "Response"
    # Access variables
    media_curve = parameters[channel]
    spend_response_df = media_curve["spend_response_df"]
    media_spend_response_data = media_curve["media_spend_response_data"]
    average_spend = media_curve["average_spending"]
    average_response = media_curve["average_response"]
    max_spend = media_curve["max_spending"]
    max_response = media_curve["max_response"]


    # Generate and Display Plot
    fig = plot_spend_response_curve(channel, spend_response_df, response_name, 
                                    average_spend, average_response, 
                                    max_spend, max_response)

    st.pyplot(fig)



elif selected_menu == "Budget Optimization":
    train_df, test_df = load_data()
    st.markdown("<h1 style='font-size: 36px;'>💰 Budget Optimization</h1>", unsafe_allow_html=True)
    st.markdown("---")  # Creates a horizontal line
    st.markdown("<h2 style='font-size: 30px;'>Response Decomposition Waterfall</h2>", unsafe_allow_html=True)


    # Load or simulate data (Replace with actual DataFrame)
    contribution_df = pd.read_csv('data/contribution_data.csv')

    # Generate the plot
    fig = plot_response_decomposition(contribution_df)

    # Display in Streamlit
    st.pyplot(fig)

    # Display Table
    budget_allocated = pd.read_csv("data/budget_allocated.csv", index_col=0)
    
    
    start_date = test_df.index[0].strftime("%Y-%m-%d")
    end_date = test_df.index[-1].strftime("%Y-%m-%d")
    period = (test_df.index[-1] - test_df.index[0]).days

    # Plot Data (Bar Chart)
    st.markdown("---")  # Creates a horizontal line
    st.subheader(f"Budget Allocation over {period} days from {start_date} to {end_date}")
    
    # Use Plotly for better visualization
    fig = px.bar(
        budget_allocated,
        x="media_channel",
        y=["average_spend", "optimal_spend"],
        barmode="group",
        title="Average vs Optimal Budget Allocation",
        labels={"value": "Spend Amount ($)", "variable": "Spend Type"},
        color_discrete_map={"average_spend": "#4c72b0", "optimal_spend": "#dd8452"}  # Custom colors
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    table = budget_allocated.copy()
    table.columns = ['Media Channels', 'Average Spend ($)', 'Optimal Spend ($)']
    st.table(table)

    ## Plot optimisation
    st.markdown("---")  # Creates a horizontal line
    st.subheader(f"Response Curve Analysis")
    with open("data/optimal_response_curve_dict.pkl", "rb") as f:
        optimal_parameters = pickle.load(f)

    # User Inputs for Spend & Response Statistics
    channel = st.selectbox("Select Media Channel:", MEDIA_CHANNELS)
    response_name = "Response"
    # Access variables
    media_curve = optimal_parameters[channel]
    spend_response_df = media_curve["spend_response_df"]
    media_spend_response_data = media_curve["media_spend_response_data"]
    average_spend = media_curve["average_spending"]
    average_response = media_curve["average_response"]
    max_spend = media_curve["max_spending"]
    max_response = media_curve["max_response"]
    optimal_spend = media_curve["optimal_spending"]
    optimal_response = media_curve["optimal_response"]


    # Generate and Display Plot
    fig = plot_spend_response_curve(channel, spend_response_df, response_name, 
                                    average_spend, average_response, 
                                    max_spend, max_response, optimal_spend, optimal_response)

    st.pyplot(fig)









