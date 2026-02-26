"""
concentration_oce.py - Concentration inequalities for OCE-RCPS

Generalizes the reference concentration.py to handle OCE risk measures.

For standard RCPS (identity cost):
    tlambda(losses, delta) = WSR_mu_plus(losses, delta) - losses.mean()
    UCB = Rhat + tlambda = R⁺

For OCE-RCPS with CVaR:
    For each candidate t, we compute UCB on the CVaR risk.
    tlambda returns the "margin" t such that UCB = Rhat + t
    
The key difference: OCE risk depends on parameter t, so we need to:
1. Generate candidate t values (via bootstrap or fixed grid)
2. For each t, compute UCB on the OCE risk
3. Take minimum over t (with Bonferroni correction)
"""

import os
import sys
import numpy as np
from scipy.stats import binom, norm
from scipy.optimize import brentq
import pickle as pkl

# Import bounds
dir_path = os.path.dirname(os.path.realpath(__file__))
from core.bounds_oce import (
    WSR_mu_plus, HB_mu_plus,
    WSR_mu_plus_cvar, HB_mu_plus_cvar,
    WSR_mu_plus_entropic,
    compute_cvar_risk, compute_entropic_risk
)


# =============================================================================
# tlambda functions for standard RCPS (from reference)
# =============================================================================

def get_tlambda_standard(bound_str, num_calib, maxiters):
    """
    Get tlambda function for STANDARD RCPS (identity cost function).
    This matches the reference implementation exactly.
    """
    bound_str = bound_str.lower()
    
    if bound_str == 'clt':
        def _tlambda(losses, delta):
            rhat = losses.mean()
            sig = losses.std()
            return -norm.ppf(delta) * sig / np.sqrt(num_calib)
        return _tlambda
    
    if bound_str == 'wsr':
        def _tlambda(losses, delta):
            # R⁺ - R̂ = t
            return WSR_mu_plus(losses, delta, maxiters) - losses.mean()
        return _tlambda
    
    raise NotImplementedError(f"Unknown bound: {bound_str}")


# =============================================================================
# tlambda functions for OCE-RCPS
# =============================================================================

def get_tlambda_oce_cvar(bound_str, beta, maxiters, num_grid=200):
    """
    Get tlambda function for OCE-RCPS with CVaR risk.
    
    For CVaR, the risk is: R_OCE(t) = t + E[max(X-t, 0)] / (1-β)
    
    We need to compute UCB on this for each candidate t.
    The tlambda function returns: UCB - empirical_risk
    
    Args:
        bound_str: 'wsr' or 'hb'
        beta: CVaR level in [0, 1)
        maxiters: max iterations for root finding
        num_grid: grid size for HB bound
    
    Returns:
        tlambda function with signature:
            tlambda(losses, t, delta) -> UCB on CVaR risk for this t
    """
    bound_str = bound_str.lower()
    
    if bound_str == 'wsr':
        def _tlambda(losses, t, delta):
            """Returns UCB on CVaR risk for single t"""
            return WSR_mu_plus_cvar(losses, t, delta, beta, maxiters)
        return _tlambda
    
    if bound_str == 'hb':
        def _tlambda(losses, t, delta):
            return HB_mu_plus_cvar(losses, t, delta, beta, num_grid, maxiters)
        return _tlambda
    
    raise NotImplementedError(f"Unknown bound: {bound_str}")


def get_tlambda_oce_entropic(bound_str, eta, alpha, maxiters):
    """
    Get tlambda function for OCE-RCPS with entropic risk.
    
    Args:
        bound_str: 'wsr' (HB not recommended for entropic due to range issues)
        eta: risk aversion parameter
        alpha: maximum t value (for normalization)
        maxiters: max iterations
    
    Returns:
        tlambda function
    """
    bound_str = bound_str.lower()
    
    if bound_str == 'wsr':
        def _tlambda(losses, t, delta):
            return WSR_mu_plus_entropic(losses, t, delta, eta, alpha, maxiters)
        return _tlambda
    
    raise NotImplementedError(f"Bound {bound_str} not supported for entropic risk")


# =============================================================================
# Bootstrap-based t generation
# =============================================================================

def generate_t_candidates_beta_region(losses_opt, num_bootstrap, beta, risk_type='cvar', eta=None):
    losses = np.asarray(losses_opt)
    n = len(losses)

    # sort losses
    L_sorted = np.sort(losses)

    # index k = ceil(n*beta)
    k = int(np.ceil(n * beta))

    # careful: Python is 0-indexed
    lower = L_sorted[k-1]
    upper = L_sorted[k] if k < n else L_sorted[-1]

    if risk_type == 'cvar':
        t_candidates = np.linspace(lower, upper, num=num_bootstrap)

    # t_candidates = np.linspace(0, 1, num=num_bootstrap)

    return t_candidates

def generate_t_candidates_bootstrap(losses_opt, num_bootstrap, beta, risk_type='cvar', eta=None):
    """
    Generate candidate t values using bootstrap from optimization set.
    
    For each bootstrap resample, find optimal t that minimizes empirical OCE risk.
    
    Args:
        losses_opt: losses from optimization dataset
        num_bootstrap: number of bootstrap resamples
        beta: CVaR level (for CVaR) or eta (for entropic)
        risk_type: 'cvar' or 'entropic'
        eta: risk aversion for entropic (required if risk_type='entropic')
    
    Returns:
        array of candidate t values
    """
    n = len(losses_opt)
    t_candidates = []
    
    for _ in range(num_bootstrap):
        # Bootstrap resample
        idx = np.random.choice(n, size=n, replace=True)
        losses_boot = losses_opt[idx]
        
        # Find optimal t for this resample
        if risk_type == 'cvar':
            _, t_opt = compute_cvar_risk(losses_boot, beta)
        else:
            _, t_opt = compute_entropic_risk(losses_boot, beta)
        
        t_candidates.append(t_opt)
    
    return np.array(t_candidates)


def generate_t_candidates_fixed(alpha, num_points):
    """Generate fixed grid of t values in [0, alpha]."""
    return np.linspace(0, alpha, num_points)


# =============================================================================
# UCB computation with Bonferroni correction over t
# =============================================================================

def compute_oce_ucb_cvar(losses, t_candidates, delta, beta, bound_str, maxiters, num_grid=200):
    """
    Compute UCB on CVaR OCE risk with Bonferroni correction over t candidates.
    
    UCB = min_{t in T} UCB(R_OCE(t), delta/|T|)
    
    Args:
        losses: calibration losses in [0, 1]
        t_candidates: array of candidate t values
        delta: confidence parameter
        beta: CVaR level
        bound_str: 'wsr' or 'hb'
        maxiters: max iterations
        num_grid: grid size for HB
    
    Returns:
        UCB on CVaR risk (minimum over t with Bonferroni)
    """
    losses = np.asarray(losses)
    all_zero = np.all(losses == 0)
    if all_zero:
        ucb_min = 0
        return ucb_min

    t_candidates = np.asarray(t_candidates)
    m = len(t_candidates)
    delta_t = delta / m  # Bonferroni correction
    
    tlambda = get_tlambda_oce_cvar(bound_str, beta, maxiters, num_grid)
    
    ucb_min = np.inf
    for t in t_candidates:
        ucb_t = tlambda(losses, t, delta_t)
        ucb_min = min(ucb_min, ucb_t)
    
    return ucb_min


def compute_oce_ucb_entropic(losses, t_candidates, delta, eta, alpha, bound_str, maxiters):
    """
    Compute UCB on entropic OCE risk with Bonferroni correction.
    """
    losses = np.asarray(losses)
    t_candidates = np.asarray(t_candidates)
    m = len(t_candidates)
    delta_t = delta / m
    
    tlambda = get_tlambda_oce_entropic(bound_str, eta, alpha, maxiters)
    
    ucb_min = np.inf
    for t in t_candidates:
        ucb_t = tlambda(losses, t, delta_t)
        ucb_min = min(ucb_min, ucb_t)
    
    return ucb_min


# =============================================================================
# Lambda selection (generalized from reference)
# =============================================================================

def get_lhat_from_table_standard(calib_loss_table, lambdas_table, gamma, delta, tlambda, bound_str):
    """
    Standard RCPS lambda selection (from reference).
    
    For identity cost function (standard expected risk).
    """
    calib_loss_table = calib_loss_table[:, ::-1]
    avg_loss = calib_loss_table.mean(axis=0)
    
    for i in range(1, len(lambdas_table)):
        Rhat = avg_loss[i]
        t = tlambda(calib_loss_table[:, i], delta)
        if (Rhat > gamma) or (Rhat + t > gamma):
            return lambdas_table[-i + 1]
    
    return lambdas_table[-1]


def get_lhat_from_table_oce(calib_loss_table, lambdas_table, gamma, delta, 
                            t_candidates_fn, ucb_fn, bound_str):
    """
    OCE-RCPS lambda selection.
    
    For each lambda, compute UCB on OCE risk using the provided UCB function.
    Select smallest lambda such that UCB <= gamma for all larger lambdas.
    
    Args:
        calib_loss_table: (n_samples, n_lambdas) array of losses
        lambdas_table: array of lambda values
        gamma: target risk level (alpha in the paper)
        delta: confidence parameter
        t_candidates_fn: function that takes losses and returns t candidates
        ucb_fn: function(losses, t_candidates, delta) -> UCB
        bound_str: bound type string
    
    Returns:
        optimal lambda_hat
    """
    # Reverse order: lambdas_table goes from small to large,
    # but we search from large to small
    calib_loss_table = calib_loss_table[:, ::-1]
    
    for i in range(1, len(lambdas_table)):
        losses_i = calib_loss_table[:, i]
        
        # Generate t candidates for this lambda
        t_candidates = t_candidates_fn(losses_i)
        
        # Compute UCB
        ucb = ucb_fn(losses_i, t_candidates, delta)
        
        if ucb > gamma:
            return lambdas_table[-i + 1]
    
    return lambdas_table[-1]


# =============================================================================
# Main interface functions
# =============================================================================

def get_bound_fn_from_string(bound_str):
    """Get bound function from string identifier."""
    bound_str = bound_str.upper()
    if bound_str == 'WSR':
        return WSR_mu_plus
    elif bound_str == 'HB':
        return HB_mu_plus
    else:
        raise NotImplementedError(f"Unknown bound: {bound_str}")


if __name__ == "__main__":
    # Test the OCE concentration
    np.random.seed(42)
    
    num_calib = 500
    delta = 0.1
    beta = 0.5
    maxiters = int(1e5)
    
    # Generate test losses
    losses = np.random.beta(2, 5, num_calib)
    print(f"Test losses: n={num_calib}, mean={losses.mean():.3f}")
    
    # Test standard RCPS
    print("\n--- Standard RCPS (identity cost) ---")
    tlambda_std = get_tlambda_standard('wsr', num_calib, maxiters)
    t_std = tlambda_std(losses, delta)
    ucb_std = losses.mean() + t_std
    print(f"R̂ = {losses.mean():.4f}, t = {t_std:.4f}, UCB = {ucb_std:.4f}")
    
    # Test CVaR RCPS
    print(f"\n--- CVaR RCPS (β={beta}) ---")
    true_cvar, t_opt = compute_cvar_risk(losses, beta)
    print(f"True CVaR: {true_cvar:.4f} (optimal t={t_opt:.4f})")
    
    # Generate t candidates via bootstrap
    t_candidates = generate_t_candidates_bootstrap(losses, 10, beta, 'cvar')
    print(f"Bootstrap t candidates: {t_candidates}")
    
    # Compute UCB
    ucb_cvar = compute_oce_ucb_cvar(losses, t_candidates, delta, beta, 'wsr', maxiters)
    print(f"UCB on CVaR: {ucb_cvar:.4f}")
    
    print("\n--- Validity check ---")
    print(f"UCB ({ucb_cvar:.4f}) >= True CVaR ({true_cvar:.4f}): {ucb_cvar >= true_cvar}")
