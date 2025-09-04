import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from lifelines.statistics import logrank_test
from warnings import warn
import random
from scipy.stats import norm

def features_normal_cov_toeplitz(n_samples, n_features: int = 30,
                                 cov_corr: float = 0.5, dtype="float64"):
    """Normal features generator with toeplitz covariance

    An example of features obtained as samples of a centered Gaussian
    vector with a toeplitz covariance matrix

    Parameters
    ----------
    n_samples : `int`, default=200
        Number of samples

    n_features : `int`, default=30
        Number of features

    cov_corr : `float`, default=0.5
        correlation coefficient of the Toeplitz correlation matrix

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the arrays used.

    Returns
    -------
    output : `numpy.ndarray`, shape=(n_samples, n_features)
        n_samples realization of a Gaussian vector with the described
        covariance

    """
    cov = toeplitz(cov_corr ** np.arange(0, n_features))
    features = np.random.multivariate_normal(
        np.zeros(n_features), cov, size=n_samples)
    if dtype != "float64":
        return features.astype(dtype)
    return features

def weights_sparse_exp(n_features: int = 100, n_active_features: int = 10, scale: float = 10.,
                       dtype="float64") -> np.ndarray:
    """Sparse and exponential model weights generator

    Instance of weights for a model, given by a vector with
    exponentially decaying components: the j-th entry is given by

    .. math: (-1)^j \exp(-j / scale)

    for 0 <= j <= nnz - 1. For j >= nnz, the entry is zero.

    Parameters
    ----------
    n_weigths : `int`, default=100
        Number of weights

    n_active_features : `int`, default=10
        Number of non-zero weights

    scale : `float`, default=10.
        The scaling of the exponential decay

    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the arrays used.

    Returns
    -------
    output : np.ndarray, shape=(n_weigths,)
        The weights vector
    """
    if n_active_features >= n_features:
        warn(("nnz must be smaller than n_weights "
              "using nnz=n_weigths instead"))
        n_active_features = n_features
    idx = np.arange(n_active_features)
    out = np.zeros(n_features, dtype=dtype)
    out[:n_active_features] = np.exp(-idx / scale)
    out[:n_active_features:2] *= -1

    return out


def compute_logrank_test(control, treat):
    """
    Perform a two-sample log-rank test comparing the survival distributions
    of control and treatment groups.

    Args:
        control (DataFrame): Subset of the dataset where treatment == 0.
        treat (DataFrame): Subset of the dataset where treatment == 1.

    Returns:
        float: Negative logarithm of the p-value from the log-rank test.
    """
    surv_time_control = control['time'].values
    surv_event_control = control['censor'].values.astype(bool)
    surv_time_treat = treat['time'].values
    surv_event_treat = treat['censor'].values.astype(bool)

    result = logrank_test(
        surv_time_control, surv_time_treat,
        event_observed_A=surv_event_control,
        event_observed_B=surv_event_treat
    )
    return -np.log(result.p_value)




def simulation(treatment_effect, n_samples, independent = True, feature_types_list = ["pos", "real", "cat"],
               n_features_bytype = 4, n_active_features = 3 , p_treated = 0.5, shape_T = 2,
               shape_C = 2, scale_C = 6., scale_C_indep = 4.5, data_types_create = True, seed=0):
    """
    Simulate a survival dataset with structured covariates and treatment effect.

    Parameters:
    -----------

    treatment_effect : float
        Coefficient for the binary treatment variable.

    n_samples : int
        Number of samples (rows) to generate.

    surv_type : str, default='surv_piecewise'
        Type identifier for the survival outcome (used in metadata output).

    n_features_bytype : int, default=4
        Number of features per feature type (real, positive, categorical).

    n_active_features : int, default=3
        Number of non-zero coefficients in beta (not directly used here).

    p_treated : float, default=0.5
        Probability of receiving treatment (for Bernoulli sampling).

    shape_T : float, default=2
        Shape parameter for the Weibull survival time distribution.

    shape_C : float, default=1
        Shape parameter for the Weibull censoring time distribution.

    scale_C : float, default=2
        Scale parameter for the censoring distribution.

    data_types_create : bool, default=True
        Whether to return a DataFrame describing feature types (metadata).

    seed : int, default=0
        Random seed for reproducibility.

    Returns:
    --------
    control : DataFrame
        Subset of the simulated dataset with untreated individuals.
    treated : DataFrame
        Subset of the simulated dataset with treated individuals.
    data_types : DataFrame (optional)
        Metadata describing variable names, types, dimensions, and classes
        (only returned if `data_types_create=True`).

    Notes:
    ------
    - Feature matrix `X` has 3 segments: real-valued, positive (abs), and binary (0/1).
    - Treatment assignment is random based on `p_treated`.
    - Survival and censoring times are simulated using Weibull models.
    - Censoring indicator: 1 = event occurred, 0 = censored.
    """

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Define feature dimensions
    n_feature_types = len(feature_types_list)
    n_features = n_feature_types * n_features_bytype
    feat_coefs = np.concatenate([weights_sparse_exp(n_features_bytype, n_active_features) for _ in range(len(feature_types_list))])
    coefs = np.insert(feat_coefs, 0, -treatment_effect)
    
    # Generate feature matrix
    X = features_normal_cov_toeplitz(n_samples, n_features)
    # Apply transformations by feature type
    for i, feat_type in enumerate(feature_types_list):
        if feat_type in ["real"]:
            continue
        elif feat_type in ["pos"]:
            X[:, i * n_features_bytype : (i + 1) * n_features_bytype] = np.abs(X[:, i * n_features_bytype : (i + 1) * n_features_bytype])
        elif feat_type in ["cat"]:
            X[:, i * n_features_bytype : (i + 1) * n_features_bytype] = 1 * (X[:, i * n_features_bytype : (i + 1) * n_features_bytype] >= 0)
        else:
            raise ValueError("Feature type {} is not supported yet".format(feat_type))
    
    # Assign treatment
    treatment = np.random.binomial(1, p_treated, size = (n_samples, 1))
    
    # Build design matrix and compute marker
    design = np.hstack((treatment, X))
    marker = np.dot(design, coefs)
    U = np.random.uniform(size = n_samples)
    V = np.random.uniform(size = n_samples)
    # Simulate survival and censoring times
    T = (-np.log(1 - U) / np.exp(marker))**(1 / shape_T)
    if independent:
        C = scale_C_indep * (-np.log(1 - V))**(1 / shape_C)
    else:
        C = scale_C * (-np.log(1 - V) / np.exp(marker))**(1 / shape_C)

    # Remove sample has survival time is zero
    mask = (T > 0) & (C > 0)
    X = X[mask]
    T = T[mask]
    C = C[mask]
    # Build final dataset
    data = pd.DataFrame(X)
    data['treatment'] = treatment
    data['time'] = np.min([T, C], axis=0)
    data['censor'] = np.argmin([C, T], axis=0)

    # Split by treatment
    control = data[data['treatment'] == 0]
    treated = data[data['treatment'] == 1]

    # Optionally create data type specification
    if data_types_create == True:

        names = []
        for x in range(1, n_feature_types * n_features_bytype + 1):
            names.append("feat{0}".format(x))
        names.append("survcens")
        
        types = np.concatenate([np.repeat(feat_type, n_features_bytype) for feat_type in feature_types_list]).tolist()
        types.append("surv")

        dims = np.repeat(1, n_feature_types * n_features_bytype).tolist()
        dims.append(2)

        nclasses = []
        for feat_type in feature_types_list:
            if feat_type in ["cat"]:
                nclasses.append(np.repeat("2", n_features_bytype))
            else:
                nclasses.append(np.repeat("", n_features_bytype))
        nclasses = np.concatenate(nclasses).tolist()
        nclasses.append("")

        data_types = pd.DataFrame({'name' : names , 'type' : types , 'dim' : dims, 'nclass' : nclasses})

        return(control, treated, data_types)

    else :
        return(control, treated)    

def cpower(mc , mi , loghaz, alpha):
    """
    mc : number of survivors in control arm
    mi : number of survivors in treated arm
    loghaz : log of hazard ratios / treatment coefficient
    alpha : level of test
    """
    ## Find its variance
    v = 1/mc + 1/mi

    ## Get same as /sasmacro/samsizc.sas if use 4/(mc+mi)

    sd = np.sqrt(v)

    z =  -norm.ppf(alpha/2)

    Power = 1 - (norm.cdf(z - np.abs(loghaz)/sd) - norm.cdf(-z - np.abs(loghaz)/sd))
    return(Power)


    
    
