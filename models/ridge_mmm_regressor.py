import pickle
import sys
sys.path.append('.')
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge 
from sklearn.model_selection import TimeSeriesSplit

from utils.plot_helpers import plot_forecast
from utils.metrics import mape_metrics, rmse_metrics
from utils.data_helpers import train_test_split_time_series
from utils.config import FEATURES, CONTROL_FEATURES, MEDIA_CHANNELS, OPTIMIZATION_PERCENTAGE, OPTUNA_TRIALS, TARGET, adstock_features_params, hill_slopes_params, hill_half_saturations_params  
from utils.prediction_helpers import budget_optimization, estimate_contribution, get_optimal_response_point, model_refit, optuna_optimize, process_response_curve


if __name__ == "__main__":
    
    #Load DATA
    model_name = "ridge"
    freq = "D"
    file_path = f"data/{freq}_preprocessed_data.csv"
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    daily_df = data.copy()
    daily_df.columns = data.columns.str.replace('_spend_raw', '', regex=True)
    #Train and test split
    train_df, test_df, start_predicted_index, end_predicted_index = train_test_split_time_series(data, test_size=0.2)
    
    #Save train_df and test_df
    train_df.to_csv("data/train_data.csv", index=True)  # Saves with datetime index
    test_df.to_csv("data/test_data.csv", index=True)  # Saves with datetime index

    tscv = TimeSeriesSplit(n_splits=2, test_size = 5)

    experiment = optuna_optimize(trials = OPTUNA_TRIALS, 
                             data = daily_df, 
                             target = TARGET, 
                             features = FEATURES + CONTROL_FEATURES, 
                             adstock_features = MEDIA_CHANNELS, 
                             adstock_features_params = adstock_features_params, 
                             hill_slopes_params = hill_slopes_params,
                             hill_half_saturations_params = hill_half_saturations_params,
                             regressor="ridge",
                             tscv = tscv)
    
    best_params = experiment.best_trial.user_attrs["params"]
    adstock_params_best = experiment.best_trial.user_attrs["adstock_alphas"]
    hill_slopes_params_best = experiment.best_trial.user_attrs["hill_slopes"]
    hill_half_saturations_params_best = experiment.best_trial.user_attrs["hill_half_saturations"]

    result = model_refit(data = daily_df, 
                        target = TARGET,
                        features = FEATURES + CONTROL_FEATURES, 
                        media_channels = MEDIA_CHANNELS,
                        organic_channels = [], 
                        model_params = best_params, 
                        adstock_params = adstock_params_best, 
                        hill_slopes_params=hill_slopes_params_best,
                        hill_half_saturations_params=hill_half_saturations_params_best,
                        regressor=model_name,
                        start_index = start_predicted_index, 
                        end_index = end_predicted_index)
    
    result['model_data'].to_csv("data/prediction_result_data.csv", index=True)
    
    #save feature coefficients 
    feature_coefficients = {}
    for feature, model_feature, coef in zip(result["features"], result["model_features"], result["model"].coef_):
        feature_coefficients[feature] = coef
        print(f"feature: {feature} -> coefficient {coef}")

    #Evaluation
    # Get evaluation data
    predictions = result["prediction_interval"]
    ridge_root_mean_squared_error = rmse_metrics(result["y_true_interval"], predictions)
    ridge_mape_error = mape_metrics(result["y_true_interval"], predictions)
    # Save prediction
    np.save(f"data/{model_name}_predictions.npy", predictions)

    print(f'Root Mean Squared Error | RMSE: {ridge_root_mean_squared_error}')
    print(f'Mean Absolute Percentage Error | MAPE: {ridge_mape_error} %')

    # # Visualisation
    # target_name = 'conversion'
    # title = 'Daily User Forecast'
    # xlabel = 'Date'
    # ylabel = 'Number of Users'
    # plot_forecast(train_df, test_df, predictions, target_name, title, xlabel, ylabel)


    spend_response_curve_dict, media_spend_response_data = process_response_curve(MEDIA_CHANNELS, result, adstock_params_best, hill_slopes_params_best, hill_half_saturations_params_best, feature_coefficients)

    contribution_df = estimate_contribution(result, media_spend_response_data, start_predicted_index, end_predicted_index)

    with open("data/spend_response_curve_dict.pkl", "wb") as f:
        pickle.dump(spend_response_curve_dict, f)

    media_spend_response_data.to_csv("data/media_spend_response_data.csv", index=True)

    #Contributions
    contribution_df.to_csv("data/contribution_data.csv", index=True)

    #Optimisation
   
    budget_allocated = budget_optimization(result, OPTIMIZATION_PERCENTAGE, feature_coefficients, hill_slopes_params_best, hill_half_saturations_params_best)
    budget_allocated.to_csv("data/budget_allocated.csv")

    ## Plot optimal response curve
    optimal_response_curve_dict = spend_response_curve_dict.copy()
    budget_allocated_values = budget_allocated['optimal_spend'].values
    optimized_spend_channels, optimized_response_channels = get_optimal_response_point(MEDIA_CHANNELS, result, budget_allocated_values, adstock_params_best, hill_slopes_params_best, 
                           hill_half_saturations_params_best, feature_coefficients)
    
    for i, channel in enumerate(MEDIA_CHANNELS):
        optimal_response_curve_dict[channel].update({
            "optimal_spending": optimized_spend_channels[i],
            "optimal_response": optimized_response_channels[i]
        })

    with open("data/optimal_response_curve_dict.pkl", "wb") as f:
        pickle.dump(optimal_response_curve_dict, f)


        




    

    
    
    
