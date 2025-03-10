import numpy as np
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from utils.metrics import rmse_metrics
from scipy import optimize
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Hyperparameter optimization
import optuna as opt
from functools import partial

import warnings
from utils.config import CONTROL_FEATURES, MEDIA_CHANNELS, TARGET
warnings.filterwarnings("ignore")

class AdstockGeometric(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: np.ndarray):
        x_decayed = np.zeros_like(X)
        x_decayed[0] = X[0]
        
        for xi in range(1, len(x_decayed)):
            x_decayed[xi] = X[xi] + self.alpha * x_decayed[xi - 1]
        return x_decayed
    
    
class HillSaturation(BaseEstimator, TransformerMixin):
    def __init__(self, slope_s, half_saturation_k):
        if slope_s < 0 or half_saturation_k < 0:
            raise ValueError("slope_s and half_saturation_k must be non-negative")
                             
        self.slope_s = slope_s
        self.half_saturation_k = half_saturation_k
        self.epsilon = 1e-9  # small constant value to avoid division by zero
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: np.ndarray, x_point = None):
        
        self.half_saturation_k_transformed  = self.half_saturation_k * (np.max(X) - np.min(X)) + np.min(X)
        
        if x_point is None:
            return (1 + self.half_saturation_k_transformed**self.slope_s / (X**self.slope_s + self.epsilon))**-1
        
        #calculate y at x_point
        return (1 + self.half_saturation_k_transformed**self.slope_s / (x_point**self.slope_s + self.epsilon))**-1

def ridge_model(ridge_alpha, x_train, y_train, x_test):
    ridge = Ridge(alpha = ridge_alpha, random_state=42)
    ridge.fit(x_train, y_train)
    prediction = ridge.predict(x_test)

    return prediction

       
# Hyperparameter Optimization
def optuna_trial(trial, 
                 data:pd.DataFrame, 
                 target,
                 features,
                 adstock_features, 
                 adstock_features_params, 
                 hill_slopes_params, 
                 hill_half_saturations_params, 
                 regressor,
                 tscv):
    
    data_temp = data.copy()
    adstock_alphas = {}
    hill_slopes = {}
    hill_half_saturations = {}

    for feature in adstock_features:
        adstock_param = f"{feature}_adstock"
        min_, max_ = adstock_features_params[adstock_param]
        adstock_alpha = trial.suggest_float(f"adstock_alpha_{feature}", min_, max_)
        adstock_alphas[feature] = adstock_alpha
        
        hill_slope_param = f"{feature}_hill_slope"
        min_, max_ = hill_slopes_params[hill_slope_param]
        hill_slope = trial.suggest_float(f"hill_slope_{feature}", min_, max_)
        hill_slopes[feature] = hill_slope
        
        hill_half_saturation_param = f"{feature}_hill_half_saturation"
        min_, max_ = hill_half_saturations_params[hill_half_saturation_param]
        hill_half_saturation = trial.suggest_float(f"hill_half_saturation_{feature}", min_, max_)
        hill_half_saturations[feature] = hill_half_saturation
        
        
        #adstock transformation
        x_feature = data[feature].values.reshape(-1, 1)
        temp_adstock = AdstockGeometric(alpha = adstock_alpha).fit_transform(x_feature)
        
        #hill saturation transformation
        temp_hill_saturation = HillSaturation(slope_s = hill_slope, half_saturation_k=hill_half_saturation).fit_transform(temp_adstock)
        data_temp[feature] = temp_hill_saturation

    if regressor == "ridge":
        #Ridge parameters
        ridge_alpha = trial.suggest_float("ridge_alpha", 0.01, 1000)
        scores = []

        #cross validation
        for train_index, test_index in tscv.split(data_temp):
            x_train = data_temp.iloc[train_index][features]
            y_train =  data_temp[target].values[train_index]
            
            x_test = data_temp.iloc[test_index][features]
            y_test = data_temp[target].values[test_index]
            

            # Put X_train and X_test into the same scale
            scale = StandardScaler()

            x_train = scale.fit_transform(x_train)
            x_test = scale.transform(x_test)
            
            #params of Ridge Regression
            params = {"alpha": ridge_alpha}
            
            prediction = ridge_model(ridge_alpha, x_train, y_train, x_test)
            rmse = rmse_metrics(test_set = y_test, predicted = prediction)
            scores.append(rmse)

    elif regressor == "prophet":
        scores = []
        # Cross-validation for Prophet
        for train_index, test_index in tscv.split(data_temp):
            # Prepare train and test sets for Prophet
            train_df = data_temp.iloc[train_index][[target] + features].reset_index()
            train_df.rename(columns={"date": "ds", target: "y"}, inplace=True)

            test_df = data_temp.iloc[test_index][features].reset_index()
            test_df.rename(columns={"date": "ds"}, inplace=True)

            # Tune Prophet Hyperparameters
            changepoint_prior_scale = trial.suggest_float("changepoint_prior_scale", 0.001, 0.5)
            seasonality_prior_scale = trial.suggest_float("seasonality_prior_scale", 0.01, 10)
            seasonality_mode = trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"])

            # Store hyperparameters in `params`
            params = {
                "changepoint_prior_scale": changepoint_prior_scale,
                "seasonality_prior_scale": seasonality_prior_scale,
                "seasonality_mode": seasonality_mode
            }

            # Initialize Prophet Model with Tuned Parameters
            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                seasonality_mode=seasonality_mode
            )
            
            # Add external regressors
            for feature in features:
                model.add_regressor(feature)
            
            # Fit Prophet Model with regressors
            model.fit(train_df)

            # Create Future DataFrame for test set & Predict
            future = test_df[['ds'] + features]  # Include external features
            forecast = model.predict(future)

            prediction = forecast['yhat'].values  # Get predicted values
            
            # Compute RMSE
            rmse = rmse_metrics(test_set=data_temp[target].iloc[test_index], predicted=prediction)
            scores.append(rmse)

        
    elif regressor == "sarimax":
        scores = []
        # Cross-validation
        for train_index, test_index in tscv.split(data_temp):
            x_train = data_temp.iloc[train_index][features]
            y_train = data_temp[target].iloc[train_index]
            
            x_test = data_temp.iloc[test_index][features]
            y_test = data_temp[target].iloc[test_index]

            if regressor == "sarimax":
                # Tune SARIMAX Hyperparameters
                p = trial.suggest_int("p", 0, 3)  # AR order
                d = trial.suggest_int("d", 0, 1)  # Differencing order
                q = trial.suggest_int("q", 0, 3)  # MA order
                P = trial.suggest_int("P", 0, 2)  # Seasonal AR order
                D = trial.suggest_int("D", 0, 1)  # Seasonal differencing order
                Q = trial.suggest_int("Q", 0, 2)  # Seasonal MA order
                s = 7  # Seasonality period

                # Store hyperparameters in `params`
                params = {
                    "p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q, "s": s
                }

                # Train SARIMAX Model
                model = SARIMAX(
                    y_train,  
                    exog=x_train,  
                    order=(p, d, q),  
                    seasonal_order=(P, D, Q, s),  
                    enforce_stationarity=False,  
                    enforce_invertibility=False
                )
                fitted_model = model.fit(disp=False)

                # Predict for test period
                prediction = fitted_model.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, exog=x_test)

            else:
                raise ValueError("Unsupported regressor. Use 'sarimax'.")

            # Compute RMSE
            rmse = rmse_metrics(test_set=y_test, predicted=prediction)
            scores.append(rmse)
        
        
    trial.set_user_attr("scores", scores)
    trial.set_user_attr("params", params)

    trial.set_user_attr("adstock_alphas", adstock_alphas)
    trial.set_user_attr("hill_slopes", hill_slopes)
    trial.set_user_attr("hill_half_saturations", hill_half_saturations)
    
    
    #average of all scores    
    return np.mean(scores)

def optuna_optimize(trials, 
                    data: pd.DataFrame, 
                    target, 
                    features,
                    adstock_features, 
                    adstock_features_params, 
                    hill_slopes_params, 
                    hill_half_saturations_params, 
                    regressor,
                    tscv, 
                    seed = 42):
    
    print(f"data size: {len(data)}")
    print(f"features: {features}")
    print(f"Model: {regressor}")

    opt.logging.set_verbosity(opt.logging.WARNING) 
    
    study_mmm = opt.create_study(direction='minimize', sampler = opt.samplers.TPESampler(seed=seed))  
        
    optimization_function = partial(optuna_trial, 
                                    data = data, 
                                    target = target, 
                                    features = features, 
                                    adstock_features = adstock_features, 
                                    adstock_features_params = adstock_features_params, 
                                    hill_slopes_params = hill_slopes_params, 
                                    hill_half_saturations_params = hill_half_saturations_params, 
                                    regressor = regressor,
                                    tscv = tscv, 
                                    )
    
    
    study_mmm.optimize(optimization_function, n_trials = trials, show_progress_bar = True)
    
    return study_mmm

#Prediction
def model_refit(data, 
                target, 
                features, 
                media_channels, 
                organic_channels,
                model_params, 
                adstock_params, 
                hill_slopes_params,
                hill_half_saturations_params,
                regressor,
                start_index, 
                end_index):
    
    data_refit = data.copy()

    best_params = model_params

    adstock_alphas = adstock_params

    #apply adstock transformation

    temporal_features = [feature if feature not in media_channels and feature not in organic_channels else f"{feature}_hill" for feature in features]

    for feature in media_channels + organic_channels:
        adstock_alpha = adstock_alphas[feature]
        print(f"applying geometric adstock transformation on {feature} with alpha {adstock_alpha:0.3}") 

        #adstock transformation
        x_feature = data_refit[feature].values.reshape(-1, 1)
        temp_adstock = AdstockGeometric(alpha = adstock_alpha).fit_transform(x_feature)

        hill_slope = hill_slopes_params[feature]
        hill_half_saturation = hill_half_saturations_params[feature]
        print(f"applying saturation hill transformation on {feature} with saturation slope {hill_slope:0.3} and half saturation {hill_half_saturation:0.3}")

        temp_hill_saturation = HillSaturation(slope_s = hill_slope, half_saturation_k=hill_half_saturation).fit_transform(temp_adstock)
        data_refit[f"{feature}_adstock"] = temp_adstock
        data_refit[f"{feature}_hill"] = temp_hill_saturation

    #build the final model on the data until the end analysis index
    x_train = data_refit.iloc[:start_index][temporal_features].copy()
    y_train = data_refit[target].values[:start_index]
    
    scale = StandardScaler()
    X_train = scale.fit_transform(x_train)

    if regressor == "ridge":
        #build ridge using the best parameters
        model = Ridge(random_state=42, **best_params)
        model.fit(X_train, y_train) 

        #concentrate on the analysis interval
        y_test = data_refit[target].values[start_index:end_index]

        #transformed data
        x_test = data_refit.iloc[start_index:end_index][temporal_features].copy()
        X_test = scale.transform(x_test)

        #conversion prediction for the analysis interval
        print(f"predicting {len(x_test)} instances")
        prediction = model.predict(X_test)

        #non transformed data set for the analysis interval 
        x_input_interval_nontransformed = data.iloc[start_index:end_index]

    elif regressor == "prophet":
        # Concentrate on the analysis interval
        y_test = data_refit[target].iloc[start_index:end_index].values  # Ensure correct slicing

        # Prepare Prophet-compatible dataframe
        x_test = data_refit.iloc[start_index:end_index][temporal_features].reset_index()
        x_test.rename(columns={"date": "ds"}, inplace=True)  # Prophet requires "ds" column for dates

        # Initialize Prophet Model (Using Best Parameters if Tuned)
        model = Prophet(
            changepoint_prior_scale=best_params["changepoint_prior_scale"],
            seasonality_prior_scale=best_params["seasonality_prior_scale"],
            seasonality_mode=best_params["seasonality_mode"]
        )

        # Add external regressors
        for feature in temporal_features:
            model.add_regressor(feature)

        # Fit the model on refitted data
        train_df = data_refit[[target] + temporal_features].reset_index()
        train_df.rename(columns={"date": "ds", target: "y"}, inplace=True)

        model.fit(train_df)
        forecast = model.predict(x_test)
        prediction = forecast["yhat"].values  # Extract Prophet predictions
        # Non-transformed dataset for the analysis interval
        x_input_interval_nontransformed = data.iloc[start_index:end_index].copy()

    if regressor == "sarimax":
        #build using the best parameters
        model = SARIMAX(
            y_train,  
            exog=x_train,  
            order=(best_params["p"], best_params["d"], best_params["q"]),  
            seasonal_order=(best_params["P"], best_params["D"], best_params["Q"], best_params["s"]),  
            enforce_stationarity=False,  
            enforce_invertibility=False
        )
        fitted_model = model.fit(disp=False)

        # Concentrate on the analysis interval
        y_test = data_refit[target].values[start_index:end_index]  # Extract ground truth

        # Transformed data for SARIMAX
        x_test = data_refit.iloc[start_index:end_index][temporal_features].copy()  # Extract exogenous features

        # Forecast for the analysis interval
        print(f"Predicting {len(x_test)} instances...")
        # Ensure model is fitted

        prediction = fitted_model.forecast(steps=len(x_test), exog=x_test)

        # Non-transformed dataset for the analysis interval
        x_input_interval_nontransformed = data_refit.iloc[start_index:end_index]


    return {
            'x_input_interval_nontransformed': x_input_interval_nontransformed, 
            'x_input_interval_transformed' : x_test,
            'prediction_interval': prediction, 
            'y_true_interval': y_test,
            'model': model,
            'model_train_data': x_train,
            'model_data': data_refit, 
            'model_features': temporal_features, 
            'features': features
           }


def process_response_curve(media_channels, result, adstock_params_best, hill_slopes_params_best, 
                           hill_half_saturations_params_best, feature_coefficients):
    """
    Processes media channels to compute spend-response relationships and store necessary data.

    Args:
        media_channels (list): List of media channels.
        result (dict): Contains model data for media channels.
        adstock_params_best (dict): Adstock parameters for each channel.
        hill_slopes_params_best (dict): Hill saturation slope parameters.
        hill_half_saturations_params_best (dict): Hill half-saturation parameters.
        feature_coefficients (dict): Model coefficients for each channel.

    Returns:
        dict: A dictionary containing spend-response data for each channel.
    """
    media_spend_response_data = []
    spend_response_curve_dict = {}

    for media_channel in media_channels:
        print(f"Processing: {media_channel}")

        # Extract parameters
        adstock = adstock_params_best[media_channel]
        hill_slope = hill_slopes_params_best[media_channel]
        hill_half_saturation = hill_half_saturations_params_best[media_channel]
        coef = feature_coefficients[media_channel]

        print(f"\tAdstock: {adstock}")
        print(f"\tSaturation Slope: {hill_slope}, Half Saturation K: {hill_half_saturation}")
        print(f"\tCoefficient: {coef}")

        # Retrieve spending data
        spendings = result["model_data"][media_channel].values
        average_nonzero_spending = int(spendings[spendings > 0].mean())
        average_spending = int(spendings.mean())
        max_spending = int(spendings.max())

        print(f"\tAverage Spend: {average_spending}, Average Non-Zero Spend: {average_nonzero_spending}")
        print(f"\tMin Spend: {spendings.min()}, Max Spend: {max_spending}")

        # Apply Adstock transformation
        spendings_adstocked = AdstockGeometric(alpha=adstock).fit_transform(spendings)

        # Calculate response values
        hill_saturation = HillSaturation(slope_s=hill_slope, half_saturation_k=hill_half_saturation)
        average_response = coef * hill_saturation.transform(X=spendings_adstocked, x_point=average_nonzero_spending)
        max_response = coef * hill_saturation.transform(X=spendings_adstocked, x_point=max_spending)

        print(f"\tAverage Response: {average_response}")
        print(f"\tMax Response: {max_response}")

        # Apply Hill Saturation Transformation
        spendings_saturated = hill_saturation.fit_transform(spendings_adstocked)
        response = spendings_saturated * coef

        # Create DataFrame
        spend_response_temp_df = pd.DataFrame({
            'spend': spendings_adstocked, 
            'response': response, 
            'media_channel': media_channel
        })

        # Store results
        media_spend_response_data.append(spend_response_temp_df)
        spend_response_curve_dict[media_channel] = {
            "spend_response_df": spend_response_temp_df,
            "media_spend_response_data": media_spend_response_data,
            "average_spending": average_nonzero_spending,
            "average_response": average_response,
            "max_spending": max_spending,
            "max_response": max_response
        }

    media_spend_response_data = pd.concat(media_spend_response_data)

    return spend_response_curve_dict, media_spend_response_data

def estimate_contribution(result, media_spend_response_data, start_predicted_index, end_predicted_index):
    
    response_df = pd.DataFrame()
    for media_channel in MEDIA_CHANNELS:
        response = media_spend_response_data[media_spend_response_data['media_channel'] == media_channel].iloc[start_predicted_index:end_predicted_index].response.values
        response_total = response.sum()
        
        response_df = pd.concat([response_df, pd.DataFrame({'media': [media_channel], 'total_effect': [response_total]})]).reset_index(drop=True)
    response_df["effect_share"] = response_df["total_effect"] / response_df["total_effect"].sum()

    organic_data = result["model_data"].iloc[start_predicted_index:end_predicted_index]
    organic_data = organic_data[[TARGET] + CONTROL_FEATURES]

    # Calculate effect share
    if CONTROL_FEATURES[0] == 'organic_proxy':
        effect_share_organic = organic_data["organic_proxy"].sum() / organic_data["conversion"].sum()

    contribution_df = pd.DataFrame({
    "channels": MEDIA_CHANNELS + CONTROL_FEATURES,
    "effect_share": list(response_df['effect_share'].values * (1 - effect_share_organic)) + [effect_share_organic] 
    })
    
    return contribution_df

# Optimisation

def budget_constraint(media_spend, budget):  
  return np.sum(media_spend) - budget


def saturation_objective_function(coefficients, 
                                  hill_slopes, 
                                  hill_half_saturations, 
                                  media_min_max_dictionary, 
                                  media_inputs):
    
    responses = []
    for i in range(len(coefficients)):
        coef = coefficients[i]
        hill_slope = hill_slopes[i]
        hill_half_saturation = hill_half_saturations[i]
        
        min_max = np.array(media_min_max_dictionary[i])
        media_input = media_inputs[i]
        
        hill_saturation = HillSaturation(slope_s = hill_slope, half_saturation_k=hill_half_saturation).transform(X = min_max, x_point = media_input)
        response = coef * hill_saturation
        responses.append(response)
        
    responses = np.array(responses)
    responses_total = np.sum(responses)
    return -responses_total

def budget_optimization(result, optimization_percentage, feature_coefficients, hill_slopes_params_best, hill_half_saturations_params_best):

    media_channel_average_spend = result["model_data"][MEDIA_CHANNELS].mean(axis=0).values

    lower_bound = media_channel_average_spend * np.ones(len(MEDIA_CHANNELS))*(1-optimization_percentage)
    upper_bound = media_channel_average_spend * np.ones(len(MEDIA_CHANNELS))*(1+ optimization_percentage)

    boundaries = optimize.Bounds(lb=lower_bound, ub=upper_bound)

    media_coefficients = [feature_coefficients[media_channel] for media_channel in MEDIA_CHANNELS]
    media_hill_slopes = [hill_slopes_params_best[media_channel] for media_channel in MEDIA_CHANNELS]
    media_hill_half_saturations = [hill_half_saturations_params_best[media_channel] for media_channel in MEDIA_CHANNELS]

    media_min_max = [(result["model_data"][f"{media_channel}_adstock"].min(),result["model_data"][f"{media_channel}_adstock"].max())  for media_channel in MEDIA_CHANNELS]

    partial_saturation_objective_function = partial(saturation_objective_function, 
                                                media_coefficients, 
                                                media_hill_slopes, 
                                                media_hill_half_saturations, 
                                                media_min_max)
    
    max_iterations = 1000
    solver_func_tolerance = 1.0e-10

    solution = optimize.minimize(
        fun=partial_saturation_objective_function,
        x0=media_channel_average_spend,
        bounds=boundaries,
        method="SLSQP",
        jac="3-point",
        options={
            "maxiter": max_iterations,
            "disp": True,
            "ftol": solver_func_tolerance,
        },
        constraints={
            "type": "eq",
            "fun": budget_constraint,
            "args": (np.sum(media_channel_average_spend), )
        })
    
    budget_allocated = pd.DataFrame({
        "media_channel": MEDIA_CHANNELS,
        "average_spend": media_channel_average_spend,
        "optimal_spend": solution.x
    })
    
    print(budget_allocated)

    return budget_allocated


def get_optimal_response_point(media_channels, result, budget_allocated_values, adstock_params_best, hill_slopes_params_best, 
                           hill_half_saturations_params_best, feature_coefficients):
    """
    Estimate optimal spend-response relationships and store necessary data.

    Args:
        media_channels (list): List of media channels.
        result (dict): Contains model data for media channels.
        solution (dict): Optimisation's result
        adstock_params_best (dict): Adstock parameters for each channel.
        hill_slopes_params_best (dict): Hill saturation slope parameters.
        hill_half_saturations_params_best (dict): Hill half-saturation parameters.
        feature_coefficients (dict): Model coefficients for each channel.

    Returns:
        optimal_spend_response_curve_dict (dict): A dictionary containing spend-response data for each channel.
        media_spend_response_data (dataframe)
    """

    optimized_spend_channels, optimized_response_channels = [], []
    #holds spend and response time series along with average spend/response for plotting spend-response curve
    for i, media_channel in enumerate(media_channels):
        print(f"Processing: {media_channel}")
        
        adstock = adstock_params_best[media_channel]
        hill_slope = hill_slopes_params_best[media_channel]
        hill_half_saturation = hill_half_saturations_params_best[media_channel]
        coef = feature_coefficients[media_channel]
        ######################################################

        # Retrieve spending data
        spendings = result["model_data"][media_channel].values
        spendings_adstocked = AdstockGeometric(alpha = adstock).fit_transform(spendings)
        #optimized
        optimized_spend = budget_allocated_values[i]
        optimized_response = coef * HillSaturation(slope_s=hill_slope, half_saturation_k=hill_half_saturation).transform(X = spendings_adstocked, x_point = optimized_spend)

        optimized_spend_channels.append(optimized_spend)
        optimized_response_channels.append(optimized_response)
        print(f"\toptimized spend: {optimized_spend:0.2f}")
        print(f"\toptimized response: {optimized_response:0.2f}")

    return optimized_spend_channels, optimized_response_channels
    







    
