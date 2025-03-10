import numpy as np
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from utils.metrics import rmse_metrics

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
        
        if regressor == "ridge":
            #params of Ridge Regression
            params = {"alpha": ridge_alpha}
        
        prediction = ridge_model(ridge_alpha, x_train, y_train, x_test)
        rmse = rmse_metrics(test_set = y_test, predicted = prediction)
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
    y_train = data[target].values[:start_index]
    
    scale = StandardScaler()
    X_train = scale.fit_transform(x_train)

    if regressor == "ridge":
        #build ridge using the best parameters
        model = Ridge(random_state=42, **best_params)
        model.fit(X_train, y_train) 


    #concentrate on the analysis interval
    y_test = data[target].values[start_index:end_index]

    #transformed data
    x_test = data_refit.iloc[start_index:end_index][temporal_features].copy()
    X_test = scale.transform(x_test)

    #conversion prediction for the analysis interval
    print(f"predicting {len(x_test)} instances")
    prediction = model.predict(X_test)

    #non transformed data set for the analysis interval 
    x_input_interval_nontransformed = data.iloc[start_index:end_index]


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


def process_media_channels(media_channels, result, adstock_params_best, hill_slopes_params_best, 
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





    
