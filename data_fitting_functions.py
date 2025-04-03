from sklearn.linear_model import LogisticRegression
from pygam import LinearGAM, LogisticGAM, s
from scipy.optimize import curve_fit, minimize
import numpy as np
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
import math

def fit_logistic_regression(x_data,y_data,x_vals_to_predict):
    ''' Fit logistic regression model using sklearn
    Inputs:
        x_data, y_data: column vectors of predictor (x) and outcome (y) variables
        x_vals_to_predict: predcitor variables for which we will generate predictions
    Returns 
        predictions: array of predicted output for all x_vals_to_predict
    '''
    # need to reshape these arrays for LogisticRegression
    x_data = np.array(x_data).reshape(-1,1); y_data = np.array(y_data).reshape(-1,1); 
    x_vals_to_predict=np.array(x_vals_to_predict).reshape(-1,1)

    # Perform logistic regression to get P(y|x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_reg = LogisticRegression(penalty='none', solver="lbfgs", max_iter=1000)
        log_reg.fit(x_data, y_data)
        # get probabilities for all x_vals_to_predict
        probs = log_reg.predict_proba(x_vals_to_predict)[:,1] # column 1 contains probability of infection

    return probs

# Define the scaled logistic function
def scaled_logistic(x, k, beta_0, beta_1):
    """
    Scaled logistic function
    
    Parameters:
    x (array-like): Input values.
    k (float): Maximum value (scale).
    beta_0 (float): Intercept parameter for linear regression
    beta_1 (float): Slope parameter.
    
    Returns:
    array-like: Scaled logistic function values.
    """
    return k / (1 + np.exp(-(beta_0 + beta_1*x)))

def neg_log_likelihood_scaled_logit(params, data):
    # unpack data
    k, beta_0, beta_1 = params
    Abs = np.array(data[0]); infected = np.array(data[1])

    # Get likelihood
    epsilon = 1e-10  # To prevent log(0)
    prob_pos = infected * np.log(np.clip(scaled_logistic(Abs, k, beta_0, beta_1), epsilon, 1 - epsilon))
    prob_neg = (1 - infected) * np.log(np.clip(1 - scaled_logistic(Abs, k, beta_0, beta_1), epsilon, 1 - epsilon))
    
    return -1*np.sum(prob_pos + prob_neg)

# # Gradient of negative log-likelihood
# def grad_nll(params, data):
#     k, beta_0, beta_1 = params
#     Abs = np.array(data[0])
#     infected = np.array(data[1])

#     # Compute logistic function
#     f_A = scaled_logistic(Abs, k, beta_0, beta_1)
#     one_minus_f_A = 1 - f_A

#     # Compute derivatives
#     dL_dk = np.sum(infected / np.clip(f_A, 1e-10, 1) - (1 - infected) / np.clip(one_minus_f_A, 1e-10, 1))
#     dL_db0 = np.sum(infected * (1 - f_A) - (1 - infected) * (-f_A))
#     dL_db1 = np.sum((infected * (1 - f_A) - (1 - infected) * (-f_A)) * Abs)

#     return -np.array([dL_dk, dL_db0, dL_db1])  # Negative because we minimize

# def fit_scaled_logistic(x_data, y_data, initial_params=[0.5, 1, -1]):

#     # Bounds: k and beta_0 must be positive, beta_1 must be negative
#     bounds = [(1e-3, 1), (0, np.inf), (-np.inf,0)]
#     # Merge data
#     data = [x_data,y_data]

#     # Optimize using L-BFGS-B
#     result = minimize(neg_log_likelihood_scaled_logit, 
#         initial_params, 
#         args=(data,),
#         jac=grad_nll,  # Supply the gradient function
#         method="L-BFGS-B",
#         bounds=bounds,
#         options={'maxiter': 5000}
#     )

#     return result.x


# Function to fit the scaled logistic regression
def fit_scaled_logistic(x_data, y_data, initial_guess=(0.5, 1, -1)):
    """
    Fit the scaled logistic regression model.
    
    Parameters:
    x_data (array-like): Independent variable values.
    y_data (array-like): Dependent variable values.
    initial_guess (tuple): Initial guesses for k, m, and s.
    
    Returns:
    tuple: Fitted parameters (k, m, s) and covariance matrix.
    """
    data = [x_data,y_data]

    result = minimize(neg_log_likelihood_scaled_logit, initial_guess, \
                    method='Nelder-Mead', args=(data), \
                    options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 10000, 'maxfev': 20000}, \
                    bounds=((0,1),(0,np.inf),(-np.inf,0))) # Force beta0 > 0 and beta1 < 0
    if result.success == False:
        print("Did not find optimal parameter fit to minimize likelihood")
    return result.x

def one_minus_OR(predicted_probabiliies):
    baseline_odds = predicted_probabiliies[0] / (1-predicted_probabiliies[0])
    VE = [1 - (pred/(1-pred))/baseline_odds for pred in predicted_probabiliies]
    return VE 

def get_mean_absolute_error(estimate,true):
    ''' Mean absolute error '''
    true = np.array(true)
    estimate = np.array(estimate)
    return np.mean(np.abs(true - estimate))

def get_L2_norm_error(estimate,true):
    '''Calculate L2 norm '''
    diff = np.array(true) - np.array(estimate)
    sum_squares = np.sum(diff**2)
    return math.sqrt(sum_squares)

def get_L1_norm_error(estimate,true):
    '''Calculate L1 norm '''
    diff = np.array(true) - np.array(estimate)
    return np.sum(abs(diff))