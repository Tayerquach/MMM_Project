# Define parameters for data generation
data_generator_params = {
    "channels": ["tv_ad", "social_ad", "search_ad"],
    "adstock_alphas": [0.50, 0.25, 0.05],  # Adstock effect parameters
    "saturation_lamdas": [1.5, 2.5, 3.5],  # Saturation effect parameters
    "betas": [350, 150, 50],  # Impact coefficients for each channel
    "spend_scalars": [10, 15, 20]  # Scaling factors for ad spend
}