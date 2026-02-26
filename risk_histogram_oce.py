"""
risk_histogram_oce.py - Experiments for OCE-RCPS

Generalizes the reference risk_histogram.py to handle OCE risk measures.

This script runs experiments comparing different bounds (WSR, HB, CLT)
for OCE-RCPS with CVaR or entropic risk.
"""
import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))   # adjust if needed
sys.path.insert(0, PROJECT_ROOT)

import os
import sys
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
from scipy.stats import norm
import seaborn as sns
from tqdm import tqdm

# Import OCE modules
from core.bounds_oce import compute_cvar_risk, compute_entropic_risk, WSR_mu_plus
from core.concentration_oce import (
    get_tlambda_standard,
    generate_t_candidates_bootstrap,
    generate_t_candidates_fixed,
    compute_oce_ucb_cvar,
    compute_oce_ucb_entropic, generate_t_candidates_beta_region
)


# =============================================================================
# Data loading (adapt to your dataset)
# =============================================================================

def get_example_loss_and_size_tables(loss_path, size_path, lambdas_table):
    """
    Load precomputed loss and size tables.
    
    Args:
        loss_path: path to loss table file
        size_path: path to size table file
        lambdas_table: array of lambda values
    
    Returns:
        loss_table: (n_samples, n_lambdas) array
        size_table: (n_samples, n_lambdas) array
    """
    # loss_table = torch.load(loss_path, weights_only=False).cpu().numpy()
    # size_table = torch.load(size_path, weights_only=False).cpu().numpy()

    loss_table = np.load(loss_path, allow_pickle=True)
    size_table = np.load(size_path, allow_pickle=True)
    return loss_table, size_table

def crc(x, beta=0.5, t=0):
    cvar_lamba = t + (1.0 / (1.0 - beta)) * np.maximum(x - t, 0.0)
    b_lamba = t + (1.0 / (1.0 - beta)) * np.maximum(0 - t, 0.0)
    n = cvar_lamba.shape[0]
    h_lamba = (cvar_lamba.sum() + b_lamba) / (n + 1)

    return h_lamba

def crc_entropic(losses, eta=5.0, t=0.0):

    x = losses

    Li_t = t + (np.exp(eta * (x - t)) - 1.0) / eta
    Bt   = t + (np.exp(eta * (0 - t)) - 1.0) / eta

    return (Li_t.sum() + Bt) / (x.shape[0] + 1.0)

# =============================================================================
# Trial functions
# =============================================================================
def trial_oce_crc_entropic(calib_losses, val_losses, val_sizes, lambdas_table,
                        opt_losses, gamma, delta, beta, bound_str,
                        num_bootstrap, maxiters):
    """
    Single trial for OCE-CRC with CVaR risk.

    Args:
        calib_losses: (n_calib, n_lambdas) calibration losses
        val_losses: (n_val, n_lambdas) validation losses
        val_sizes: (n_val, n_lambdas) validation sizes
        lambdas_table: array of lambda values
        opt_losses: (n_opt, n_lambdas) optimization losses for bootstrap
        gamma: target risk level (alpha)
        delta: confidence parameter
        beta: CVaR level
        bound_str: 'wsr' or 'hb'
        num_bootstrap: number of bootstrap resamples
        maxiters: max iterations for bounds

    Returns:
        lhat, risk, sizes
    """
    # Search from right to left
    calib_losses_rev = calib_losses[:, ::-1]
    opt_losses_rev = opt_losses[:, ::-1]

    lhat_idx = len(lambdas_table) - 1

    for i in range(1, len(lambdas_table)):
        losses_calib_i = calib_losses_rev[:, i]
        losses_opt_i = opt_losses_rev[:, i]

        # Generate t candidates from optimization set
        t_candidates = generate_t_candidates_bootstrap(
            losses_opt_i, 1, beta, 'entropic'
        )

        # Compute risk on calibration set
        estimate_risk = crc_entropic(
            losses_calib_i, beta, t_candidates
        )

        if estimate_risk <= gamma:
            lhat_idx = len(lambdas_table) - i


    lhat = lambdas_table[lhat_idx]

    # Evaluate TRUE CVaR risk on validation set
    val_losses_at_lhat = val_losses[:, lhat_idx]
    true_cvar, _ = compute_entropic_risk(val_losses_at_lhat, beta)
    sizes = val_sizes[:, lhat_idx]

    return lhat, true_cvar, sizes


def trial_oce_crc_cvar(calib_losses, val_losses, val_sizes, lambdas_table,
                        opt_losses, gamma, delta, beta, bound_str,
                        num_bootstrap, maxiters):
    """
    Single trial for OCE-CRC with CVaR risk.

    Args:
        calib_losses: (n_calib, n_lambdas) calibration losses
        val_losses: (n_val, n_lambdas) validation losses
        val_sizes: (n_val, n_lambdas) validation sizes
        lambdas_table: array of lambda values
        opt_losses: (n_opt, n_lambdas) optimization losses for bootstrap
        gamma: target risk level (alpha)
        delta: confidence parameter
        beta: CVaR level
        bound_str: 'wsr' or 'hb'
        num_bootstrap: number of bootstrap resamples
        maxiters: max iterations for bounds

    Returns:
        lhat, risk, sizes
    """
    # Search from right to left
    calib_losses_rev = calib_losses[:, ::-1]
    opt_losses_rev = opt_losses[:, ::-1]

    lhat_idx = len(lambdas_table) - 1

    for i in range(1, len(lambdas_table)):
        losses_calib_i = calib_losses_rev[:, i]
        losses_opt_i = opt_losses_rev[:, i]

        # Generate t candidates from optimization set
        t_candidates = generate_t_candidates_bootstrap(
            losses_opt_i, 1, beta, 'cvar'
        )

        # Compute risk on calibration set
        estimate_risk = crc(
            losses_calib_i, beta, t_candidates
        )

        if estimate_risk <= gamma:
            lhat_idx = len(lambdas_table) - i


    lhat = lambdas_table[lhat_idx]

    # Evaluate TRUE CVaR risk on validation set
    val_losses_at_lhat = val_losses[:, lhat_idx]
    true_cvar, _ = compute_cvar_risk(val_losses_at_lhat, beta)
    sizes = val_sizes[:, lhat_idx]

    return lhat, true_cvar, sizes

def trial_standard_rcps(calib_losses, val_losses, val_sizes, lambdas_table, 
                        gamma, delta, tlambda):
    """
    Single trial for standard RCPS (identity cost function).
    
    Returns:
        lhat: selected lambda
        risk: mean loss on validation set at lhat
        sizes: sizes on validation set at lhat
    """
    # Find lambda_hat using calibration data
    # Search from right to left (large to small lambda)
    calib_losses_rev = calib_losses[:, ::-1]
    
    lhat_idx = len(lambdas_table) - 1  # Default to largest lambda
    for i in range(1, len(lambdas_table)):
        losses_i = calib_losses_rev[:, i]
        Rhat = losses_i.mean()
        t = tlambda(losses_i, delta)
        if (Rhat > gamma) or (Rhat + t > gamma):
            lhat_idx = len(lambdas_table) - i
            break
    
    lhat = lambdas_table[lhat_idx]
    
    # Evaluate on validation set
    risk = val_losses[:, lhat_idx].mean()
    sizes = val_sizes[:, lhat_idx]
    
    return lhat, risk, sizes


def trial_oce_rcps_cvar(calib_losses, val_losses, val_sizes, lambdas_table,
                        opt_losses, gamma, delta, beta, bound_str, benchmark,
                        num_bootstrap, maxiters):
    """
    Single trial for OCE-RCPS with CVaR risk.
    
    Args:
        calib_losses: (n_calib, n_lambdas) calibration losses
        val_losses: (n_val, n_lambdas) validation losses
        val_sizes: (n_val, n_lambdas) validation sizes
        lambdas_table: array of lambda values
        opt_losses: (n_opt, n_lambdas) optimization losses for bootstrap
        gamma: target risk level (alpha)
        delta: confidence parameter
        beta: CVaR level
        bound_str: 'wsr' or 'hb'
        num_bootstrap: number of bootstrap resamples
        maxiters: max iterations for bounds
    
    Returns:
        lhat, risk, sizes
    """
    # Search from right to left
    calib_losses_rev = calib_losses[:, ::-1]
    opt_losses_rev = opt_losses[:, ::-1]

    
    lhat_idx = len(lambdas_table) - 1
    
    for i in range(1, len(lambdas_table)):
        losses_calib_i = calib_losses_rev[:, i]
        losses_opt_i = opt_losses_rev[:, i]
        
        # Generate t candidates from optimization set
        if benchmark == 0:
            t_candidates = generate_t_candidates_bootstrap(
                losses_opt_i, num_bootstrap, beta, 'cvar'
            )
        else:
            t_candidates = generate_t_candidates_beta_region(
                losses_opt_i, num_bootstrap, beta, 'cvar'
            )
            losses_calib_i = np.concatenate((losses_calib_i, losses_opt_i), axis=0)
        
        # Compute UCB on calibration set
        ucb = compute_oce_ucb_cvar(
            losses_calib_i, t_candidates, delta, beta, bound_str, maxiters
        )
        
        if ucb > gamma:
            lhat_idx = len(lambdas_table) - i
            break
    
    lhat = lambdas_table[lhat_idx]
    
    # Evaluate TRUE CVaR risk on validation set
    val_losses_at_lhat = val_losses[:, lhat_idx]
    true_cvar, _ = compute_cvar_risk(val_losses_at_lhat, beta)
    sizes = val_sizes[:, lhat_idx]
    
    return lhat, true_cvar, sizes


def trial_oce_rcps_entropic(calib_losses, val_losses, val_sizes, lambdas_table,
                            opt_losses, gamma, delta, eta, alpha, bound_str,
                            num_bootstrap, maxiters):
    """
    Single trial for OCE-RCPS with entropic risk.
    """
    calib_losses_rev = calib_losses[:, ::-1]
    opt_losses_rev = opt_losses[:, ::-1]
    
    lhat_idx = len(lambdas_table) - 1
    
    for i in range(1, len(lambdas_table)):
        losses_calib_i = calib_losses_rev[:, i]
        losses_opt_i = opt_losses_rev[:, i]
        
        t_candidates = generate_t_candidates_bootstrap(
            losses_opt_i, num_bootstrap, eta, 'entropic', eta=eta
        )
        
        ucb = compute_oce_ucb_entropic(
            losses_calib_i, t_candidates, delta, eta, alpha, bound_str, maxiters
        )
        
        if ucb > gamma:
            lhat_idx = len(lambdas_table) - i
            break
    
    lhat = lambdas_table[lhat_idx]
    
    val_losses_at_lhat = val_losses[:, lhat_idx]
    true_entropic, _ = compute_entropic_risk(val_losses_at_lhat, eta)
    sizes = val_sizes[:, lhat_idx]
    
    return lhat, true_entropic, sizes

def random_split(ids, ratio):
    rng = np.random.default_rng()
    ids = np.array(ids)
    rng.shuffle(ids)
    n = int(np.floor(ratio * len(ids)))
    return ids[:n], ids[n:]
# =============================================================================
# Experiment runner
# =============================================================================

def run_experiment(loss_table, size_table, lambdas_table, 
                   gamma, delta, num_trials, num_calib, opt_split_ratio,
                   risk_type, bound_str, benchmark, beta=0.5, eta=1.0, alpha=0.4,
                   num_bootstrap=10, maxiters=int(1e5)):
    """
    Run OCE-RCPS experiment.
    
    Args:
        loss_table: (n_total, n_lambdas) precomputed losses
        size_table: (n_total, n_lambdas) precomputed sizes
        lambdas_table: array of lambda values
        gamma: target risk level
        delta: confidence parameter
        num_trials: number of random trials
        num_calib: calibration set size
        opt_split_ratio: optimization set size (for bootstrap)
        risk_type: 'standard', 'cvar', or 'entropic'
        bound_str: 'wsr', 'hb', or 'clt'
        beta: CVaR level (for CVaR)
        eta: risk aversion (for entropic)
        alpha: max t value (for entropic normalization)
        num_bootstrap: number of bootstrap resamples
        maxiters: max iterations for bounds
    
    Returns:
        DataFrame with results
    """
    n_total = loss_table.shape[0]
    
    results = []
    
    for trial in tqdm(range(num_trials), desc=f"{risk_type}-{bound_str}"):
        # Random permutation

        #######################  Random permutation option 1: sampling from overall dataset  #######################
        perm = np.random.permutation(n_total)
        num_opt = int(opt_split_ratio * num_calib)
        # Split data
        opt_idx = perm[:num_opt]
        calib_idx = perm[num_opt:num_calib]
        val_idx = perm[num_calib:]


        #######################  Random permutation option 2: sampling from each sub-dataset  #######################
        # id_ClinicDB, id_Kvasir, id_HyperKvasir, id_ColonDB, id_LaribPolypDB = list(range(0, 61)), list(range(61, 161)), list(range(161, 1161)), list(range(1161, 1541)), list(range(1542, 1737))
        # dataset_split_ratio = num_calib / n_total
        #
        # #######################  Create the reference dataset  #######################
        # id_ClinicDB_ref = np.random.choice(id_ClinicDB, int(np.floor(dataset_split_ratio * len(id_ClinicDB))), replace=False)
        # id_ClinicDB_opt, id_ClinicDB_cal = random_split(id_ClinicDB_ref, opt_split_ratio)
        #
        # id_Kvasir_ref = np.random.choice(id_Kvasir, int(np.floor(dataset_split_ratio * len(id_Kvasir))), replace=False)
        # id_Kvasir_opt, id_Kvasir_cal = random_split(id_Kvasir_ref, opt_split_ratio)
        #
        # id_HyperKvasir_ref = np.random.choice(id_HyperKvasir, int(np.floor(dataset_split_ratio * len(id_HyperKvasir))), replace=False)
        # id_HyperKvasir_opt, id_HyperKvasir_cal = random_split(id_HyperKvasir_ref, opt_split_ratio)
        #
        # id_ColonDB_ref = np.random.choice(id_ColonDB, int(np.floor(dataset_split_ratio * len(id_ColonDB))), replace=False)
        # id_ColonDB_opt, id_ColonDB_cal = random_split(id_ColonDB_ref, opt_split_ratio)
        #
        # id_LaribPolypDB_ref = np.random.choice(id_LaribPolypDB, int(np.floor(dataset_split_ratio * len(id_LaribPolypDB))), replace=False)
        # id_LaribPolypDB_opt, id_LaribPolypDB_cal = random_split(id_LaribPolypDB_ref, opt_split_ratio)
        #
        # id_ref = np.asarray([*id_ClinicDB_ref, *id_Kvasir_ref, *id_HyperKvasir_ref, *id_ColonDB_ref, *id_LaribPolypDB_ref])
        #
        # opt_idx = np.asarray([*id_ClinicDB_opt, *id_Kvasir_opt, *id_HyperKvasir_opt, *id_ColonDB_opt, *id_LaribPolypDB_opt])
        # calib_idx = np.asarray([*id_ClinicDB_cal, *id_Kvasir_cal, *id_HyperKvasir_cal, *id_ColonDB_cal, *id_LaribPolypDB_cal])
        # val_idx = np.setdiff1d(np.arange(0, n_total), id_ref)


        opt_losses = loss_table[opt_idx]
        calib_losses = loss_table[calib_idx]
        val_losses = loss_table[val_idx]
        val_sizes = size_table[val_idx]
        
        # Run trial based on risk type
        if risk_type == 'standard':
            tlambda = get_tlambda_standard(bound_str, num_calib, maxiters)
            lhat, risk, sizes = trial_standard_rcps(
                calib_losses, val_losses, val_sizes, lambdas_table,
                gamma, delta, tlambda
            )
        elif risk_type == 'cvar':
            lhat, risk, sizes = trial_oce_rcps_cvar(
                calib_losses, val_losses, val_sizes, lambdas_table,
                opt_losses, gamma, delta, beta, bound_str, benchmark,
                num_bootstrap, maxiters
            )
        elif risk_type == 'crc_cvar':
            lhat, risk, sizes = trial_oce_crc_cvar(
                calib_losses, val_losses, val_sizes, lambdas_table,
                opt_losses, gamma, delta, beta, bound_str,
                num_bootstrap, maxiters
            )
        elif risk_type == 'crc_entropic':
            lhat, risk, sizes = trial_oce_crc_entropic(
                calib_losses, val_losses, val_sizes, lambdas_table,
                opt_losses, gamma, delta, beta, bound_str,
                num_bootstrap, maxiters
            )
        elif risk_type == 'entropic':
            lhat, risk, sizes = trial_oce_rcps_entropic(
                calib_losses, val_losses, val_sizes, lambdas_table,
                opt_losses, gamma, delta, eta, alpha, bound_str,
                num_bootstrap, maxiters
            )
        else:
            raise ValueError(f"Unknown risk type: {risk_type}")
        
        results.append({
            'trial': trial,
            'bound': bound_str,
            'risk_type': risk_type,
            'lhat': lhat,
            'risk': risk,
            'sizes': sizes,
            'gamma': gamma,
            'delta': delta,
            'satisfied': risk <= gamma
        })
    
    return pd.DataFrame(results)


# =============================================================================
# Plotting
# =============================================================================

def plot_histograms(dfs, gamma, delta, output_path):
    """Plot risk and size histograms."""
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    
    for df in dfs:
        label = f"{df['risk_type'].iloc[0]}-{df['bound'].iloc[0]}"
        
        # Risk histogram
        risks = df['risk'].values
        axs[0].hist(risks, alpha=0.6, density=True, label=label, bins=30)
        
        # Size histogram
        all_sizes = np.concatenate([s for s in df['sizes'].values])
        axs[1].hist(all_sizes, alpha=0.6, density=True, label=label, bins=30)
    
    axs[0].axvline(x=gamma, c='red', linestyle='--', label=f'γ={gamma}')
    axs[0].set_xlabel('Risk')
    axs[0].set_ylabel('Density')
    axs[0].legend()
    axs[0].set_title('Risk Distribution')
    
    axs[1].set_xlabel('Set Size')
    axs[1].set_ylabel('Density')
    axs[1].set_yscale('log')
    axs[1].legend()
    axs[1].set_title('Size Distribution')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_summary(df, gamma, delta, beta):
    """Print summary statistics."""
    risk_type = df['risk_type'].iloc[0]
    bound = df['bound'].iloc[0]
    
    sat_rate = df['satisfied'].mean()
    mean_risk = df['risk'].mean()
    mean_size = np.mean([s.mean() for s in df['sizes'].values])
    median_size = np.median([s for s in df['sizes'].values])

    print(f"\n{risk_type.upper()} - {bound.upper()}:")
    print(f"  Satisfaction rate: {sat_rate:.3f} (target: >= {1-delta:.2f})")
    print(f"  Mean risk: {mean_risk:.3f} (target: <= {gamma:.2f} with beta_concentration = {beta:.2f})")
    print(f"  Mean size: {mean_size:.2f}")
    print(f"  Median size: {median_size:.2f}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCE-RCPS Experiments')
    parser.add_argument('--loss_path', type=str, default='./.cache/-1.0_0.0_1000_example_loss_table.npy', help='Path to loss table')
    parser.add_argument('--size_path', type=str, default='./.cache/-1.0_0.0_1000_example_size_table.npy', help='Path to size table')
    parser.add_argument('--output_dir', type=str, default='./outputs/', help='Output directory')
    parser.add_argument('--num_trials', type=int, default=1000, help='Number of trials')
    parser.add_argument('--num_calib', type=int, default=1000, help='Calibration set size')
    parser.add_argument('--opt_split_ratio', type=float, default=0.2, help='Optimization set size')
    parser.add_argument('--gamma', type=float, default=0.4, help='Target risk level')
    parser.add_argument('--delta', type=float, default=0.2, help='Confidence parameter')
    parser.add_argument('--beta', type=float, default=3, help='CVaR level')
    parser.add_argument('--eta', type=float, default=3.0, help='Entropic risk aversion')
    parser.add_argument('--num_bootstrap', type=int, default=1, help='Bootstrap resamples')
    parser.add_argument('--benchmark', type=int, default=0, help='0 for bootstrapping, 1 for beta-quantile region of loss')
    parser.add_argument('--bounds', type=str, nargs='+', default=['WSR'], help='Bounds to test')
    parser.add_argument('--risk_types', type=str, nargs='+', default=['crc_entropic', 'entropic'], help='Risk types')
    args = parser.parse_args()
    
    # Setup
    sns.set(palette='pastel', font='serif')
    sns.set_style('white')
    np.random.seed(42)
    
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    maxiters = int(1e5)
    lambdas_table = np.linspace(-1, 0, 1000)
    
    # Load data
    print("Loading data...")
    loss_table, size_table = get_example_loss_and_size_tables(
        args.loss_path, args.size_path, lambdas_table
    )
    print(f"Data shape: {loss_table.shape}")
    
    # Run experiments
    all_dfs = []
    
    for risk_type in args.risk_types:
        for bound_str in args.bounds:
            # if risk_type in ['crc_cvar'] and bound_str in ['WSR']:
            #     break
            print(f"\nRunning {risk_type} - {bound_str}...")
            
            df = run_experiment(
                loss_table, size_table, lambdas_table,
                gamma=args.gamma,
                delta=args.delta,
                num_trials=args.num_trials,
                num_calib=args.num_calib,
                opt_split_ratio=args.opt_split_ratio,
                risk_type=risk_type,
                bound_str=bound_str,
                benchmark=args.benchmark,
                beta=args.beta,
                eta=args.eta,
                alpha=args.gamma,  # Use gamma as max t
                num_bootstrap=args.num_bootstrap,
                maxiters=maxiters,
            )
            
            print_summary(df, args.gamma, args.delta, args.beta)
            all_dfs.append(df)
            
            # Save individual results
            fname = f"{risk_type}_{bound_str}_{args.gamma}_{args.delta}.pkl"
            df.to_pickle(os.path.join(args.output_dir, fname))
    
    # Plot
    plot_path = os.path.join(args.output_dir, f"histograms_{args.gamma}_{args.delta}.pdf")
    plot_histograms(all_dfs, args.gamma, args.delta, plot_path)
    print(f"\nPlot saved to {plot_path}")
