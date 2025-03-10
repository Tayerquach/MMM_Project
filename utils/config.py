# Define parameters for data generation
data_generator_params = {
    "channels": ["tv_ad", "social_ad", "search_ad"],
    "adstock_alphas": [0.50, 0.25, 0.05],  # Adstock effect parameters
    "saturation_lamdas": [1.5, 2.5, 3.5],  # Saturation effect parameters
    "betas": [350, 150, 50],  # Impact coefficients for each channel
    "spend_scalars": [10, 15, 20]  # Scaling factors for ad spend
}

TARGET = "conversion"
MEDIA_CHANNELS = ["tv_ad", "social_ad", "search_ad"]
CONTROL_FEATURES = ["organic_proxy"]
# control_features = ["Trend", "Seasonal"]
FEATURES = MEDIA_CHANNELS
OPTIMIZATION_PERCENTAGE = 0.2

adstock_features_params = {
    "tv_ad_adstock": (0.1, 0.4),
    "social_ad_adstock": (0.1, 0.4),
    "search_ad_adstock": (0.1, 0.4),
}
hill_slopes_params = {
    "tv_ad_hill_slope": (0.1, 5.0),
    "social_ad_hill_slope": (0.1, 5.0),
    "search_ad_hill_slope": (0.1, 5.0),
}
hill_half_saturations_params = {
    "tv_ad_hill_half_saturation": (0.1, 1.0),
    "social_ad_hill_half_saturation": (0.1, 1.0),
    "search_ad_hill_half_saturation": (0.1, 1.0),
}
OPTUNA_TRIALS = 1000
