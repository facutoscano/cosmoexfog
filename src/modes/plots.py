"""
Plotting utilities for CosmoExFog
===================================
Can be called standalone or imported by the mode modules.

Usage:
    python -m src.modes.plots --mode multipole_cuts --config config.yaml
    python -m src.modes.plots --mode discrepancy    --config config.yaml
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ── Shared style ──────────────────────────────────────────────────────────────

PARAM_LABELS = {
    'H0':        r'$H_0$ [km/s/Mpc]',
    'omegamh2':  r'$\Omega_m h^2$',
    'ombh2':     r'$\Omega_b h^2$',
    'omch2':     r'$\Omega_c h^2$',
    'ns':        r'$n_s$',
    'tau':       r'$\tau$',
    'As':        r'$A_s$',
}

PLANCK_PR3 = {
    'H0': 67.36, 'omegamh2': 0.1430, 'ombh2': 0.02237,
    'omch2': 0.1200, 'ns': 0.9649, 'tau': 0.0544,
}


# ── Mode 2 plots: parameter shift grids ──────────────────────────────────────

def plot_parameter_grid(results_file, params=None, plots_dir='.', planck=None):
    """
    Heatmap grid of best-fit parameter values across all cuts.
    Rows = simulations, columns = cuts.
    """
    df = pd.read_csv(results_file, sep='\t', comment='#')
    if params is None:
        params = ['H0', 'omegamh2']
    if planck is None:
        planck = PLANCK_PR3

    # Order cuts sensibly: full_spectrum first, then lmax ascending, lmin ascending
    cut_order = (['full_spectrum']
                 + sorted([c for c in df['cut'].unique() if c.startswith('lmax_')],
                           key=lambda x: int(x.split('_')[1]))
                 + sorted([c for c in df['cut'].unique() if c.startswith('lmin_')],
                           key=lambda x: int(x.split('_')[1])))
    cut_order = [c for c in cut_order if c in df['cut'].unique()]

    sims = sorted(df['sim'].unique())
    os.makedirs(plots_dir, exist_ok=True)

    for param in params:
        pivot = df.pivot_table(index='sim', columns='cut', values=param,
                               aggfunc='first')
        pivot = pivot.reindex(columns=cut_order, fill_value=np.nan)

        fig, ax = plt.subplots(figsize=(max(10, len(cut_order) * 0.5), 6))
        cmap = cm.RdBu_r
        ref  = planck.get(param, pivot.values[~np.isnan(pivot.values)].mean())
        vmax = np.nanmax(np.abs(pivot.values - ref)) * 1.2
        im = ax.imshow(pivot.values, aspect='auto', cmap=cmap,
                       vmin=ref - vmax, vmax=ref + vmax)
        ax.set_xticks(range(len(cut_order)))
        ax.set_xticklabels(cut_order, rotation=70, ha='right', fontsize=7)
        ax.set_yticks(range(len(sims)))
        ax.set_yticklabels(sims, fontsize=6)
        ax.set_xlabel('Cut')
        ax.set_ylabel('Simulation')
        ax.set_title(f'{PARAM_LABELS.get(param, param)} — best-fit across cuts')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(im, cax=cax, label=PARAM_LABELS.get(param, param))
        plt.tight_layout()
        fname = os.path.join(plots_dir, f'Grid_{param}.pdf')
        fig.savefig(fname, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {fname}')


def plot_shift_grid(results_file, params=None, plots_dir='.', planck=None,
                    perc_threshold=3.0):
    """
    Highlight sims/cuts where the parameter shifts by more than
    `perc_threshold` percent relative to the full-spectrum value.
    """
    df = pd.read_csv(results_file, sep='\t', comment='#')
    if params is None:
        params = ['H0', 'omegamh2']
    if planck is None:
        planck = PLANCK_PR3

    full = df[df['cut'] == 'full_spectrum'].set_index('sim')
    os.makedirs(plots_dir, exist_ok=True)

    for param in params:
        cut_order = sorted([c for c in df['cut'].unique() if c != 'full_spectrum'])
        sims = sorted(df['sim'].unique())

        shift_matrix = np.full((len(sims), len(cut_order)), np.nan)
        for j, cut in enumerate(cut_order):
            sub = df[df['cut'] == cut].set_index('sim')
            for i, sim in enumerate(sims):
                if sim in sub.index and sim in full.index:
                    ref_val = full.loc[sim, param]
                    if ref_val != 0:
                        shift_matrix[i, j] = (sub.loc[sim, param] - ref_val) / abs(ref_val) * 100

        fig, ax = plt.subplots(figsize=(max(10, len(cut_order) * 0.5), 6))
        vmax = max(np.nanmax(np.abs(shift_matrix)), perc_threshold) * 1.1
        im = ax.imshow(shift_matrix, aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax)
        # Mark cells that exceed threshold
        it = np.nditer(shift_matrix, flags=['multi_index'])
        while not it.finished:
            val = float(it[0])
            if not np.isnan(val) and abs(val) > perc_threshold:
                r, c = it.multi_index
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                            fill=False, edgecolor='gold',
                                            linewidth=1.5))
            it.iternext()

        ax.set_xticks(range(len(cut_order)))
        ax.set_xticklabels(cut_order, rotation=70, ha='right', fontsize=7)
        ax.set_yticks(range(len(sims)))
        ax.set_yticklabels(sims, fontsize=6)
        ax.set_xlabel('Cut')
        ax.set_ylabel('Simulation')
        ax.set_title(f'{PARAM_LABELS.get(param, param)} — % shift from full-spectrum')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.1)
        plt.colorbar(im, cax=cax, label='% shift')
        plt.tight_layout()
        fname = os.path.join(plots_dir, f'ShiftGrid_{param}.pdf')
        fig.savefig(fname, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {fname}')


def plot_mask_comparison(file_baseline, file_mask, param, plots_dir='.'):
    """
    Scatter plot of param(baseline) vs param(mask) for each (sim, cut).
    Points away from the diagonal indicate mask-induced shifts.
    """
    df_base = pd.read_csv(file_baseline, sep='\t', comment='#')
    df_mask = pd.read_csv(file_mask,     sep='\t', comment='#')

    merged = df_base[['sim', 'cut', param]].merge(
        df_mask[['sim', 'cut', param]], on=['sim', 'cut'],
        suffixes=('_base', '_mask'))
    if merged.empty:
        print(f'No matching rows for {param}. Skipping comparison plot.')
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(merged[f'{param}_base'], merged[f'{param}_mask'],
                    c=range(len(merged)), cmap='viridis', alpha=0.6, s=15)
    lims = [min(merged[f'{param}_base'].min(), merged[f'{param}_mask'].min()),
            max(merged[f'{param}_base'].max(), merged[f'{param}_mask'].max())]
    ax.plot(lims, lims, 'k--', lw=0.8, label='1:1')
    ax.set_xlabel(f'{PARAM_LABELS.get(param, param)} — baseline')
    ax.set_ylabel(f'{PARAM_LABELS.get(param, param)} — with mask')
    ax.set_title(f'Mask effect on {param}')
    ax.legend()
    plt.colorbar(sc, ax=ax, label='(sim, cut) index')
    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fname = os.path.join(plots_dir, f'MaskComparison_{param}.pdf')
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fname}')


# ── Mode 3 plots (also called from discrepancy_sims.py) ──────────────────────

def plot_discrepancy_histogram(delta_file, param, lsplit, sigma_threshold,
                               plots_dir='.'):
    """Histogram of Δparam with flagging lines."""
    df = pd.read_csv(delta_file, sep='\t', comment='#')
    col = f'delta_{param}'
    if col not in df.columns:
        return

    sigma = df[col].std()
    thresh = sigma_threshold * sigma

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[col], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0,      color='gray', lw=1, ls='--')
    ax.axvline(+thresh, color='red', lw=1.5, ls=':', label=f'+{sigma_threshold}σ')
    ax.axvline(-thresh, color='red', lw=1.5, ls=':',  label=f'-{sigma_threshold}σ')
    n_flagged = (np.abs(df[col]) > thresh).sum()
    ax.set_xlabel(f'Δ{PARAM_LABELS.get(param, param)}  (ℓ_split={lsplit})')
    ax.set_ylabel('Number of simulations')
    ax.set_title(f'{n_flagged}/{len(df)} sims flagged (σ={sigma:.4f})')
    ax.legend()
    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    fname = os.path.join(plots_dir, f'Histogram_Delta{param}_lsplit{lsplit}.pdf')
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fname}')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CosmoExFog plotting utilities')
    parser.add_argument('--config',  default='config.yaml')
    parser.add_argument('--mode',    choices=['multipole_cuts', 'discrepancy'],
                        required=True)
    parser.add_argument('--params',  nargs='+', default=['H0', 'omegamh2'])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    paths_cfg = config['paths']
    plots_dir = paths_cfg['plots_folder']

    if args.mode == 'multipole_cuts':
        cfg = config['multipole_cuts']
        for mask in cfg.get('mask_variants', ['']):
            cls_name = cfg['cls_name'] + mask
            results_file = os.path.join(
                paths_cfg['output_folder'], 'multipole_cuts',
                cls_name, f'Params_Iminuit_{cls_name}.txt')
            if not os.path.exists(results_file):
                print(f'[SKIP] {results_file} not found')
                continue
            plot_parameter_grid(results_file, args.params,
                                os.path.join(plots_dir, 'multipole_cuts', cls_name))
            plot_shift_grid(results_file, args.params,
                            os.path.join(plots_dir, 'multipole_cuts', cls_name))

    elif args.mode == 'discrepancy':
        cfg = config['discrepancy']
        sigma_threshold = cfg.get('sigma_threshold', 1.2)
        out_base = os.path.join(paths_cfg['output_folder'], 'discrepancy')
        for param in args.params:
            for lsplit in cfg.get('lsplit_values', [800]):
                delta_file = os.path.join(
                    out_base,
                    f"Delta_{param}_lsplit{lsplit}_{cfg['cls_name']}.txt")
                if not os.path.exists(delta_file):
                    print(f'[SKIP] {delta_file} not found')
                    continue
                plot_discrepancy_histogram(
                    delta_file, param, lsplit, sigma_threshold,
                    os.path.join(plots_dir, 'discrepancy'))