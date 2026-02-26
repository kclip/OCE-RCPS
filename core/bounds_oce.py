"""
bounds_oce.py - Concentration bounds for OCE risk measures

Generalizes the reference bounds.py to handle:
1. Standard expected risk (identity cost function) - original RCPS
2. CVaR risk 
3. Entropic risk

The key insight: For OCE risk R_OCE = inf_t { t + E[φ(X-t)] }, we need UCB on E[φ(X-t)].
For CVaR, φ(u) = max(u,0)/(1-β), and max(X-t, 0) ∈ [0,1] when X ∈ [0,1].
So we can apply standard WSR to z = max(X-t, 0) directly.
"""

import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq


# =============================================================================
# Helper functions (from reference bounds.py)
# =============================================================================

def h1(y, mu):
    """KL divergence helper"""
    return y * np.log(y / mu) + (1 - y) * np.log((1 - y) / (1 - mu))


def h2(y):
    """Bennett helper"""
    return (1 + y) * np.log(1 + y) - y


# =============================================================================
# Log tail inequalities (from reference bounds.py)
# =============================================================================

def hoeffding_plus(mu, x, n):
    return -n * h1(np.maximum(mu, x), mu)


def bentkus_plus(mu, x, n):
    return np.log(max(binom.cdf(np.floor(n * x), n, mu), 1e-10)) + 1


# =============================================================================
# Standard UCB functions (from reference bounds.py)
# =============================================================================

def HB_mu_plus(muhat, sigmahat, n, delta, num_grid, maxiters):
    """Hoeffding-Bentkus UCB on mean"""
    def _tailprob(mu):
        hoeffding_mu = hoeffding_plus(mu, muhat, n)
        bentkus_mu = bentkus_plus(mu, muhat, n)
        return min(hoeffding_mu, bentkus_mu) - np.log(delta)
    
    if _tailprob(1 - 1e-10) > 0:
        return 1
    else:
        return brentq(_tailprob, muhat, 1 - 1e-10, maxiter=maxiters)


def WSR_mu_plus(x, delta, maxiters):
    """
    Wealth process-based Sequential Ratio UCB on E[x] for x in [0,1].
    
    This is the EXACT implementation from the reference bounds.py.
    Returns mu such that P(E[x] <= mu) >= 1 - delta.
    """
    n = x.shape[0]
    muhat = (np.cumsum(x) + 0.5) / (1 + np.array(range(1, n + 1)))
    sigma2hat = (np.cumsum((x - muhat) ** 2) + 0.25) / (1 + np.array(range(1, n + 1)))
    sigma2hat[1:] = sigma2hat[:-1]
    sigma2hat[0] = 0.25
    nu = np.minimum(np.sqrt(2 * np.log(1 / delta) / n / sigma2hat), 1)
    
    def _Kn(mu):
        return np.max(np.cumsum(np.log(1 - nu * (x - mu)))) + np.log(delta)
    
    if _Kn(1) < 0:
        return 1
    return brentq(_Kn, 1e-10, 1 - 1e-10, maxiter=maxiters)


# =============================================================================
# OCE Risk: CVaR
# =============================================================================

def WSR_mu_plus_cvar(losses, t, delta, beta, maxiters):
    """
    WSR UCB for CVaR OCE risk at a SINGLE value of t.
    
    CVaR_β(X) = t + E[max(X-t, 0)] / (1-β)
    
    For z = max(X-t, 0):
    - z ∈ [0, 1] when X ∈ [0, 1] and t ≥ 0
    - Apply WSR to get UCB on E[z]
    - Return t + UCB(E[z]) / (1-β)
    
    Args:
        losses: array of losses in [0, 1]
        t: single value of parameter t
        delta: confidence parameter (already adjusted for Bonferroni if needed)
        beta: CVaR level in [0, 1)
        maxiters: max iterations for root finding
    
    Returns:
        UCB on CVaR risk for this t
    """
    z = np.maximum(losses - t, 0.0)
    mu_plus_z = WSR_mu_plus(z, delta, maxiters)
    return t + mu_plus_z / (1.0 - beta)


def HB_mu_plus_cvar(losses, t, delta, beta, num_grid, maxiters):
    """
    Hoeffding-Bentkus UCB for CVaR OCE risk at a SINGLE value of t.
    """
    z = np.maximum(losses - t, 0.0)
    n = len(z)
    muhat = z.mean()
    sigmahat = z.std()
    mu_plus_z = HB_mu_plus(muhat, sigmahat, n, delta, num_grid, maxiters)
    return t + mu_plus_z / (1.0 - beta)


# =============================================================================
# OCE Risk: Entropic
# =============================================================================

def WSR_mu_plus_entropic(losses, t, delta, eta, alpha, maxiters):
    """
    WSR UCB for entropic OCE risk at a SINGLE value of t.
    
    Entropic risk = t + E[φ(X-t)] where φ(u) = (exp(η*u) - 1) / η
    
    φ(u) is not bounded in [0,1], so we normalize using deterministic bounds:
    - u ∈ [-α, 1] when X ∈ [0,1] and t ∈ [0, α]
    - φ_min = (exp(-η*α) - 1) / η
    - φ_max = (exp(η) - 1) / η
    
    Args:
        losses: array of losses in [0, 1]
        t: single value of parameter t (in [0, α])
        delta: confidence parameter
        eta: risk aversion parameter (η > 0)
        alpha: maximum value of t (for normalization bounds)
        maxiters: max iterations
    
    Returns:
        UCB on entropic risk for this t
    """
    # Compute φ values
    u = losses - t
    phi = (np.exp(eta * u) - 1.0) / eta
    
    # Deterministic bounds for normalization
    phi_min = (np.exp(-eta * alpha) - 1.0) / eta
    phi_max = (np.exp(eta * 1.0) - 1.0) / eta
    phi_range = phi_max - phi_min
    
    # Normalize to [0, 1]
    z = (phi - phi_min) / phi_range
    z = np.clip(z, 0.0, 1.0)
    
    # Apply WSR to normalized values
    mu_plus_normalized = WSR_mu_plus(z, delta, maxiters)
    
    # Map back to original scale
    phi_ucb = phi_min + phi_range * mu_plus_normalized
    
    return t + phi_ucb


# =============================================================================
# Compute true OCE risk (for evaluation)
# =============================================================================

def compute_cvar_risk(losses, beta):
    """
    Compute true CVaR risk: inf_t { t + E[max(X-t, 0)] / (1-β) }
    
    The optimal t* is the β-quantile of the loss distribution.
    """
    losses = np.asarray(losses)
    t_opt = np.quantile(losses, beta, method="lower")
    cvar = t_opt + np.mean(np.maximum(losses - t_opt, 0)) / (1 - beta)
    return cvar, t_opt


def compute_entropic_risk(losses, eta):
    """
    Compute true entropic risk using log-mean-exp.
    """
    losses = np.asarray(losses)
    z = eta * losses
    m = np.max(z)
    log_mean_exp = m + np.log(np.mean(np.exp(z - m)))
    R_hat = log_mean_exp / eta
    return R_hat, R_hat  # For entropic, t* = R


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    losses = np.random.beta(2, 5, 1000)
    
    print("Testing bounds_oce.py")
    print(f"Loss range: [{losses.min():.3f}, {losses.max():.3f}]")
    print(f"Loss mean: {losses.mean():.3f}")
    
    # Standard WSR
    ucb_standard = WSR_mu_plus(losses, 0.1, int(1e5))
    print(f"\nStandard WSR UCB on E[X]: {ucb_standard:.4f}")
    
    # CVaR WSR
    beta = 0.5
    t = 0.2
    ucb_cvar = WSR_mu_plus_cvar(losses, t, 0.1, beta, int(1e5))
    true_cvar, t_opt = compute_cvar_risk(losses, beta)
    print(f"\nCVaR (β={beta}):")
    print(f"  True CVaR: {true_cvar:.4f} (t*={t_opt:.4f})")
    print(f"  UCB at t={t}: {ucb_cvar:.4f}")
