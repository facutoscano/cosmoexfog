"""
MODE 3: Discrepancy in Simulations
====================================
Systematic search for anomalous simulations (|delta_param| > threshold)
and re-evaluation with alternate masks to test whether the gap closes or grows.

Steps:
  1. Load iMinuit results for the baseline dataset (all sims x all cuts).
  2. For each (parameter, ell_split), compute the discrepancy
       delta_param = param(lmax<split) - param(lmin>split)
  3. Flag sims where |delta_param| > sigma_threshold (delta_param across sims).
  4. Re-run iMinuit for flagged sims using the alternate mask datasets.
  5. Compare delta_param before and after the mask to see if the gap closes.
  6. Save summary tables and plots.

Usage (from run_pipeline.py):
    python run_pipeline.py discrepancy
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from astropy.io import fits
from src.utils.data_utils import build_ell_edges_and_slice_data

warnings.filterwarnings('ignore')


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_iminuit_results(filepath):
    """Load a tab-separated iMinuit results file into a DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"iMinuit results not found: {filepath}")
    return pd.read_csv(filepath, sep='\t', comment='#')


def _load_cls_matrix(cls_path, sim_indices=None):
    """Load Cl FITS file. Returns (cl_matrix, cl_std, sim_idxs)."""
    with fits.open(cls_path) as hdul:
        data = hdul[0].data
    cl_std = np.std(data[1:], axis=0)
    idxs = list(sim_indices) if sim_indices is not None else list(range(len(data)))
    return data[idxs], cl_std, idxs


# ── Core discrepancy logic ────────────────────────────────────────────────────

def _compute_discrepancy(df, param, lsplit):
    """
    Compute delta_param = param(lmax_split) - param(lmin_split) for each sim.

    Returns a DataFrame with columns: sim, delta_{param}, lsplit.
    """
    low  = df[df['cut'] == f'lmax_{lsplit}'][['sim', param]].rename(
               columns={param: f'{param}_low'})
    high = df[df['cut'] == f'lmin_{lsplit}'][['sim', param]].rename(
               columns={param: f'{param}_high'})
    merged = low.merge(high, on='sim')
    merged[f'delta_{param}'] = merged[f'{param}_low'] - merged[f'{param}_high']
    merged['lsplit'] = lsplit
    return merged


def _flag_anomalies(delta_df, param, sigma_threshold):
    """
    Flag sims where |delta_param| > sigma_threshold.

    Returns (flagged_df, sigma_val, threshold_val).
    """
    col    = f'delta_{param}'
    sigma  = delta_df[col].std()
    thresh = sigma_threshold * sigma
    flagged = delta_df[np.abs(delta_df[col]) > thresh].copy()
    flagged['flagged'] = True
    delta_df = delta_df.copy()
    delta_df['flagged'] = np.abs(delta_df[col]) > thresh
    return delta_df, sigma, thresh


# ── Re-fitting flagged sims with alternate mask ───────────────────────────────

def _fit_one_iminuit(args):
    """Fit one (sim, cut) with iMinuit. Identical to multipole_cuts version."""
    import camb
    from iminuit import Minuit

    cl_data, cl_std, ell_bins_edges, ref_params, sim_idx, cut_name = args

    n_bins = len(cl_data)
    if n_bins < 5:
        return None

    def chi2(H0, ombh2, omch2, ns, tau, As, a_ps):
        try:
            pars = camb.CAMBparams()
            pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2,
                               tau=tau, mnu=0.06, omk=0)
            pars.InitPower.set_params(As=As * np.exp(-2 * tau), ns=ns)
            pars.set_for_lmax(int(ell_bins_edges[-1]) + 50, lens_potential_accuracy=0)
            results = camb.get_results(pars)
            cls = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total']
            theo = cls[:, 0]
            ls = np.arange(len(theo))
            theo_dl = theo + a_ps * ls * (ls + 1) / (3000. * 3001.)
            theo_bin = np.array([
                np.mean(theo_dl[int(ell_bins_edges[i]):int(ell_bins_edges[i + 1])])
                for i in range(n_bins)
            ])
            return np.sum(((theo_bin - cl_data) / cl_std) ** 2)
        except Exception:
            return 1e10

    p = ref_params.copy()
    m = Minuit(chi2,
               H0=p['H0'], ombh2=p['ombh2'], omch2=p['omch2'],
               ns=p['ns'], tau=p['tau'], As=p['As'], a_ps=p.get('a_ps', 10.0))
    m.errordef = Minuit.LEAST_SQUARES
    m.limits['H0']    = (50, 90)
    m.limits['ombh2'] = (0.018, 0.028)
    m.limits['omch2'] = (0.08, 0.16)
    m.limits['ns']    = (0.85, 1.05)
    m.limits['tau']   = (0.02, 0.12)
    m.limits['As']    = (1e-9, 4e-9)
    m.limits['a_ps']  = (0.0, 50.0)
    m.migrad()

    if not m.valid:
        return None

    omegamh2 = m.values['omch2'] + m.values['ombh2']
    return {
        'sim': sim_idx, 'cut': cut_name,
        'H0': m.values['H0'], 'omegamh2': omegamh2,
        'ombh2': m.values['ombh2'], 'omch2': m.values['omch2'],
        'ns': m.values['ns'], 'tau': m.values['tau'],
        'As': m.values['As'], 'a_ps': m.values['a_ps'],
        'min_cost': m.fval, 'valid': m.valid,
    }


def _refit_flagged(flagged_sims, cls_path, ell_bins, lsplit_values, ref_params, n_workers, mask_label, ell_min_file, ell_min, delta_ell):
    """Re-fit the flagged simulations for the given mask dataset."""
    with fits.open(cls_path) as hdul:
        data_full = hdul[0].data
    cl_std_full = np.std(data_full[1:], axis=0)

    data, cl_std, ell_bins_check = build_ell_edges_and_slice_data(data_full,cl_std_full, ell_min_file, ell_min, delta_ell)

    assert np.array_equal(ell_bins_check, ell_bins), (
        "ell_bins mismatch between baseline and refit. Aborting.")

    cuts_needed = []
    for lsplit in lsplit_values:
        cuts_needed += [
            {'name': f'lmax_{lsplit}', 'ell_cut': lsplit, 'low_cut': False},
            {'name': f'lmin_{lsplit}', 'ell_cut': lsplit, 'low_cut': True},
        ]

    tasks = []
    for sim_idx in flagged_sims:
        if sim_idx >= len(data):
            continue
        cl = data[sim_idx]
        for cut in cuts_needed:
            centers = (ell_bins[:-1] + ell_bins[1:]) / 2
            if cut['low_cut']:
                mask = centers >= cut['ell_cut']
            else:
                mask = centers <= cut['ell_cut']
            idx = np.where(mask)[0]
            if len(idx) < 5:
                continue
            cl_cut  = cl[mask]
            std_cut = cl_std[mask]
            edges_cut = ell_bins[idx[0]:idx[-1] + 2]
            tasks.append((cl_cut, std_cut, edges_cut, ref_params,
                          sim_idx, cut['name']))

    print(f"  Re-fitting {len(tasks)} tasks for mask '{mask_label}' "
          f"with {n_workers} workers ...")
    with mp.Pool(n_workers) as pool:
        results = pool.map(_fit_one_iminuit, tasks)

    return pd.DataFrame([r for r in results if r is not None])


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_discrepancy_comparison(before_df, after_dfs, param, lsplit,
                                 sigma_threshold, plots_dir, mask_labels):
    """
    Plot delta_param distribution before and after mask re-evaluation.
    Saves one figure per (param, lsplit).
    """
    col = f'delta_{param}'
    fig, axes = plt.subplots(1, 1 + len(after_dfs),
                              figsize=(5 * (1 + len(after_dfs)), 4),
                              sharey=True)
    if not hasattr(axes, '__len__'):
        axes = [axes]

    all_dfs  = [before_df] + after_dfs
    all_labels = ['Baseline'] + [f'Mask: {m}' for m in mask_labels]
    colors   = ['steelblue', 'tomato', 'seagreen', 'orchid']

    for ax, df_plot, label, color in zip(axes, all_dfs, all_labels, colors):
        sigma = df_plot[col].std()
        thresh = sigma_threshold * sigma
        flagged = df_plot[np.abs(df_plot[col]) > thresh]
        normal  = df_plot[np.abs(df_plot[col]) <= thresh]

        ax.scatter(normal['sim'],  normal[col],  c=color,   alpha=0.6,
                   s=20, label='Normal')
        ax.scatter(flagged['sim'], flagged[col], c='black',  alpha=0.9,
                   s=40, marker='*', label=f'Flagged (>{sigma_threshold:.1f}σ)')
        ax.axhline(0,      color='gray',  lw=0.8, ls='--')
        ax.axhline(+thresh, color='red',  lw=1.0, ls=':', alpha=0.7)
        ax.axhline(-thresh, color='red',  lw=1.0, ls=':', alpha=0.7)
        ax.set_xlabel('Simulation index')
        ax.set_ylabel(f'Δ{param}' if ax is axes[0] else '')
        ax.set_title(f'{label}\nσ={sigma:.4f}, thresh={thresh:.4f}')
        ax.legend(fontsize=7)

    fig.suptitle(f'Δ{param}  |  ℓ_split = {lsplit}', fontweight='bold')
    plt.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    fname = os.path.join(plots_dir, f'Discrepancy_{param}_lsplit{lsplit}.pdf')
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved → {fname}")


def _plot_gap_summary(summary_df, param, plots_dir):
    """Bar plot: fraction of flagged sims whose gap closes after each mask."""
    fig, ax = plt.subplots(figsize=(7, 4))
    masks = [c for c in summary_df.columns if c.startswith('gap_closed_')]
    x = np.arange(len(summary_df))
    width = 0.8 / max(len(masks), 1)
    for i, col in enumerate(masks):
        label = col.replace('gap_closed_', 'Mask: ')
        ax.bar(x + i * width, summary_df[col] * 100, width=width,
               label=label, alpha=0.8)
    ax.set_xticks(x + width * (len(masks) - 1) / 2)
    ax.set_xticklabels([f'ℓ_split={v}' for v in summary_df['lsplit']])
    ax.set_ylabel('% flagged sims where gap closes')
    ax.set_title(f'Gap closure after mask re-evaluation — {param}')
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fname = os.path.join(plots_dir, f'GapClosure_{param}.pdf')
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"  Summary plot saved → {fname}")


# ── Public entry point ────────────────────────────────────────────────────────

def run(args, config):
    """Entry point called by run_pipeline.py for mode 'discrepancy'."""
    cfg       = config['discrepancy']
    paths_cfg = config['paths']

    cls_name       = cfg['cls_name']
    cls_name_check = cfg['cls_name_check']
    ell_min        = cfg.get('ell_min', 32)
    ell_max        = cfg.get('ell_max', 2000)
    delta_ell      = cfg.get('delta_ell', 30)
    ell_min_file = cfg.get('ell_min_file', 2)
    cls_path_for_edges = os.path.join(paths_cfg['cls_folder'], f"{cls_name}.fits")
    with fits.open(cls_path_for_edges) as hdul:
        data = hdul[0].data
    cl_std_full = np.std(data[1:], axis=0)
    _, _, ell_bins = build_ell_edges_and_slice_data(data, cl_std_full, ell_min_file, ell_min, delta_ell)
    print(f">> ell_bins: {len(ell_bins)} edges | {ell_bins[0]}..{ell_bins[-1]}")
    
    sigma_threshold = cfg.get('sigma_threshold', 1.2)
    params_to_flag = cfg.get('params_to_flag', ['H0', 'omegamh2'])
    lsplit_values  = cfg.get('lsplit_values', [800])
    reeval_masks   = cfg.get('reeval_masks', ['', '_mask3'])
    n_workers      = cfg.get('n_workers', 4)
    ref_params     = config.get('multipole_cuts', {}).get(
                         'planck_pr3',
                         {'H0': 67.36, 'ombh2': 0.02237, 'omch2': 0.12,
                          'ns': 0.9649, 'As': 2.1e-9, 'tau': 0.0544, 'a_ps': 10.0})

    out_base   = os.path.join(paths_cfg['output_folder'], 'discrepancy')
    plots_dir  = os.path.join(paths_cfg['plots_folder'],  'discrepancy')
    os.makedirs(out_base,  exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ── Load baseline iMinuit results (or run them) ──
    baseline_file = os.path.join(
        paths_cfg['output_folder'], 'multipole_cuts',
        cls_name, f'Params_Iminuit_{cls_name}.txt')

    if os.path.exists(baseline_file):
        print(f">> Loading baseline results: {baseline_file}")
        df_base = _load_iminuit_results(baseline_file)
    else:
        print(f">> Baseline iMinuit file not found. Running iMinuit first ...")
        from src.modes.multipole_cuts import _run_iminuit as _mc_iminuit
        mc_cfg = config.get('multipole_cuts', {})
        mc_cfg.setdefault('cls_name', cls_name)
        _mc_iminuit(mc_cfg, paths_cfg, '', out_base)
        df_base = _load_iminuit_results(
            os.path.join(out_base, f'Params_Iminuit_{cls_name}.txt'))

    # ── Compute discrepancy and flag anomalies ──
    all_flagged_sims = set()
    summary_rows = []

    for param in params_to_flag:
        print(f"\n── Param: {param} ──")
        param_summary = []

        for lsplit in lsplit_values:
            delta_df, sigma, thresh = _flag_anomalies(
                _compute_discrepancy(df_base, param, lsplit),
                param, sigma_threshold)

            flagged_sims = set(delta_df[delta_df['flagged']]['sim'].tolist())
            all_flagged_sims |= flagged_sims
            n_flagged = len(flagged_sims)
            print(f"  lsplit={lsplit}: {n_flagged} flagged "
                  f"(σ={sigma:.4f}, thresh={thresh:.4f})")

            # Save the discrepancy table
            out_delta = os.path.join(
                out_base, f'Delta_{param}_lsplit{lsplit}_{cls_name}.txt')
            delta_df.to_csv(out_delta, sep='\t', index=False, float_format='%.6f')

            row = {'param': param, 'lsplit': lsplit, 'n_flagged': n_flagged,
                   'sigma': sigma, 'threshold': thresh}

            # ── Re-fit flagged sims with alternate masks ──
            after_dfs   = []
            mask_labels = []

            for mask_suffix in reeval_masks:
                if mask_suffix == '':
                    df_refit = delta_df[delta_df['sim'].isin(flagged_sims)].copy()
                    after_dfs.append(delta_df)
                    mask_labels.append('baseline')
                    continue

                cls_check_file = os.path.join(
                    paths_cfg['cls_folder'],
                    f'{cls_name}{mask_suffix}.fits')
                if not os.path.exists(cls_check_file):
                    print(f"  [SKIP] mask FITS not found: {cls_check_file}")
                    continue

                df_refit = _refit_flagged(
                    list(flagged_sims), cls_check_file, ell_bins,
                    [lsplit], ref_params, n_workers, mask_suffix,
                    ell_min_file, ell_min, delta_ell)

                if df_refit.empty:
                    continue

                # Compute delta_param again for the re-fitted sims
                delta_refit = _compute_discrepancy(df_refit, param, lsplit)
                delta_refit['flagged'] = np.abs(
                    delta_refit[f'delta_{param}']) > thresh

                n_closed = (~delta_refit['flagged']).sum()
                frac_closed = n_closed / max(len(delta_refit), 1)
                row[f'gap_closed_{mask_suffix}'] = frac_closed
                print(f"    Mask '{mask_suffix}': {n_closed}/{len(delta_refit)} "
                      f"gaps close ({frac_closed:.1%})")

                # Save re-fitted delta table
                out_refit = os.path.join(
                    out_base,
                    f'Delta_{param}_lsplit{lsplit}_{cls_name}{mask_suffix}_refit.txt')
                delta_refit.to_csv(out_refit, sep='\t', index=False,
                                   float_format='%.6f')

                after_dfs.append(delta_refit)
                mask_labels.append(mask_suffix)

            # ── Plot comparison ──
            if len(after_dfs) > 0:
                _plot_discrepancy_comparison(
                    delta_df, after_dfs[1:], param, lsplit,
                    sigma_threshold, plots_dir, mask_labels[1:])

            summary_rows.append(row)

        param_df = pd.DataFrame([r for r in summary_rows if r['param'] == param])
        if not param_df.empty:
            _plot_gap_summary(param_df, param, plots_dir)

    # ── Saving ──
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(out_base, 'Discrepancy_Summary.txt')
        summary_df.to_csv(summary_path, sep='\t', index=False, float_format='%.6f')
        print(f"\n>> Summary saved → {summary_path}")

    n_total_flagged = len(all_flagged_sims)
    print(f"\n>> MODE 3 completed. "
          f"Total unique flagged sims across all (param, lsplit): {n_total_flagged}")
    if n_total_flagged > 0:
        print(f"   Flagged sim indices: {sorted(all_flagged_sims)}")