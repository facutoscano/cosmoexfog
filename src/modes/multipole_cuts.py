"""
MODE 2: Multipole Cuts & Parameter Shifts
==========================================
Estimates cosmological parameters (H0, Ωmh², ...) using iMinuit or Cobaya
while sweeping lmin/lmax cuts and comparing multiple mask variants.

Usage (from run_pipeline.py):
    python run_pipeline.py multipole_cuts --sampler iminuit
    python run_pipeline.py multipole_cuts --sampler cobaya
"""

import os
import glob
import warnings
import numpy as np
from astropy.io import fits
import multiprocessing as mp
from functools import partial

warnings.filterwarnings('ignore')

# ── Helpers ─────────────────────────────────────────────────────────────────

def _load_cls(cls_path, ell_bins_edges):
    """Load binned Cl spectrum from FITS. Returns (cl_data, cl_std) for sim0."""
    with fits.open(cls_path) as hdul:
        data = hdul[0].data  # shape: (n_sims, n_bins) or (n_bins,)
    if data.ndim == 2:
        cl_data = data[0]    # sim0 = data
        cl_std  = np.std(data[1:], axis=0)
    else:
        cl_data = data
        cl_std  = np.ones_like(data)
    return cl_data, cl_std


def _load_all_cls(cls_path, sim_indices=None):
    """Load all simulations + data from a FITS file.

    Returns:
        cl_matrix : (n_sims, n_bins) — all sims including data at index 0
        cl_std    : (n_bins,) — std across sims [1:]
        sim_idx   : list of int — which sim indices are included
    """
    with fits.open(cls_path) as hdul:
        data = hdul[0].data  # (n_total, n_bins)
    cl_std = np.std(data[1:], axis=0)
    if sim_indices is not None:
        idxs = list(sim_indices)
    else:
        idxs = list(range(len(data)))
    return data[idxs], cl_std, idxs


def _slice_spectrum(cl, std, ell_bins_edges, ell_cut, low_cut):
    """Apply a lmin or lmax cut to the spectrum."""
    if ell_cut is None:
        return cl.copy(), std.copy(), ell_bins_edges.copy()
    centers = (ell_bins_edges[:-1] + ell_bins_edges[1:]) / 2
    mask = centers >= ell_cut if low_cut else centers <= ell_cut
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.array([]), np.array([]), np.array([])
    return cl[mask], std[mask], ell_bins_edges[idx[0]:idx[-1] + 2]


def _build_cuts(cfg):
    """Build the list of (cut_name, ell_cut, low_cut) from config."""
    cuts = [{'name': 'full_spectrum', 'ell_cut': None, 'low_cut': False}]
    for l in cfg.get('lmin_cuts', []):
        cuts.append({'name': f'lmin_{l}', 'ell_cut': l, 'low_cut': True})
    for l in cfg.get('lmax_cuts', []):
        cuts.append({'name': f'lmax_{l}', 'ell_cut': l, 'low_cut': False})
    return cuts


# ── iMinuit engine ───────────────────────────────────────────────────────────

def _fit_one_iminuit(args):
    """Fit a single (sim, cut) combination. Designed for multiprocessing.Pool."""
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
            theo = cls[:, 0]  # TT
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
        'sim': sim_idx,
        'cut': cut_name,
        'H0': m.values['H0'],
        'omegamh2': omegamh2,
        'ombh2': m.values['ombh2'],
        'omch2': m.values['omch2'],
        'ns': m.values['ns'],
        'tau': m.values['tau'],
        'As': m.values['As'],
        'a_ps': m.values['a_ps'],
        'min_cost': m.fval,
        'valid': m.valid,
    }


def _run_iminuit(cfg, paths_cfg, mask_suffix, output_dir):
    """Run iMinuit scan for all sims x all cuts for a given mask variant."""
    import pandas as pd

    cls_name  = cfg['cls_name'] + mask_suffix
    cls_file  = os.path.join(paths_cfg['cls_folder'], f'{cls_name}.fits')
    if not os.path.exists(cls_file):
        print(f"  [SKIP] FITS not found: {cls_file}")
        return

    ell_min   = cfg.get('ell_min', 32)
    ell_max   = cfg.get('ell_max', 2000)
    delta_ell = cfg.get('delta_ell', 30)
    ell_bins  = np.arange(ell_min, ell_max + delta_ell, delta_ell)

    cl_matrix, cl_std, sim_idxs = _load_all_cls(cls_file, cfg.get('sim_indices'))
    cuts      = _build_cuts(cfg)
    ref_params = cfg['planck_pr3']

    tasks = []
    for sim_idx, cl in zip(sim_idxs, cl_matrix):
        for cut in cuts:
            cl_cut, std_cut, edges_cut = _slice_spectrum(
                cl, cl_std, ell_bins, cut['ell_cut'], cut['low_cut'])
            if len(cl_cut) >= 5:
                tasks.append((cl_cut, std_cut, edges_cut, ref_params,
                               sim_idx, cut['name']))

    n_workers = cfg.get('iminuit', {}).get('n_workers', 4)
    print(f"  Running {len(tasks)} fits with {n_workers} workers ...")

    with mp.Pool(n_workers) as pool:
        results = pool.map(_fit_one_iminuit, tasks)

    rows = [r for r in results if r is not None]
    out_path = os.path.join(output_dir, f'Params_Iminuit_{cls_name}.txt')
    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep='\t', index=False,
              header=True, float_format='%.6f')
    print(f"  Saved → {out_path}")
    return df


# ── Cobaya engine ────────────────────────────────────────────────────────────

def _run_cobaya(cfg, paths_cfg, mask_suffix, output_dir):
    """Run Cobaya MCMC for each cut for the data spectrum (sim0)."""
    from cobaya.run import run as cobaya_run

    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    cls_name = cfg['cls_name'] + mask_suffix
    cls_file = os.path.join(paths_cfg['cls_folder'], f'{cls_name}.fits')
    if not os.path.exists(cls_file):
        print(f"  [SKIP] FITS not found: {cls_file}")
        return

    ell_min   = cfg.get('ell_min', 32)
    ell_max   = cfg.get('ell_max', 2000)
    delta_ell = cfg.get('delta_ell', 30)
    ell_bins  = np.arange(ell_min, ell_max + delta_ell, delta_ell)

    cl_data, cl_std = _load_cls(cls_file, ell_bins)
    cuts      = _build_cuts(cfg)
    cobaya_cfg = cfg.get('cobaya', {})
    mcmc_cfg   = {
        'Rminus1_stop': cobaya_cfg.get('Rminus1_stop', 0.05),
        'max_tries':    cobaya_cfg.get('max_tries', 10000),
    }

    for cut in cuts:
        cl_cut, std_cut, edges_cut = _slice_spectrum(
            cl_data, cl_std, ell_bins, cut['ell_cut'], cut['low_cut'])

        if len(cl_cut) < 5:
            continue

        out_prefix = os.path.join(output_dir, cls_name, cut['name'], 'mcmc_chain')
        if os.path.exists(out_prefix + '.updated.yaml') and rank == 0:
            print(f"  [SKIP] already done: {cut['name']}")
            continue

        n_bins = len(cl_cut)

        def _make_likelihood(data, std, edges, nb):
            def my_likelihood(_self, a_ps, **kwargs):
                import camb
                lmax_req = int(edges[-1])
                pars_cmb = _self.provider.get_param('H0')  # trigger CAMB
                cls_cmb  = _self.provider.get_Cl(ell_factor=True, units='muK2')
                theo = cls_cmb['tt'][:lmax_req + 1]
                ls   = np.arange(len(theo))
                theo_dl = theo + a_ps * ls * (ls + 1) / (3000. * 3001.)
                theo_bin = np.array([
                    np.mean(theo_dl[int(edges[i]):int(edges[i + 1])])
                    for i in range(nb)
                ])
                return -0.5 * np.sum(((theo_bin - data) / std) ** 2)
            return my_likelihood

        likelihood_fn = _make_likelihood(cl_cut, std_cut, edges_cut, n_bins)

        info = {
            'likelihood': {
                'my_cl_likelihood': {
                    'external': likelihood_fn,
                    'requires': {'Cl': {'tt': int(edges_cut[-1])}, 'H0': None},
                    'params': {
                        'a_ps': {'prior': {'min': 0, 'max': 50},
                                 'ref': 10.0, 'proposal': 1.0, 'latex': 'a_{ps}'}
                    }
                }
            },
            'theory': {'camb': {'extra_args': {'lens_potential_accuracy': 0}}},
            'params': {
                'H0':    {'prior': {'min': 50, 'max': 90},   'ref': 67.4, 'proposal': 0.5},
                'ombh2': {'prior': {'min': 0.018, 'max': 0.028}, 'ref': 0.0224, 'proposal': 0.0002},
                'omch2': {'prior': {'min': 0.08, 'max': 0.16},   'ref': 0.120,  'proposal': 0.002},
                'ns':    {'prior': {'min': 0.85, 'max': 1.05},   'ref': 0.965,  'proposal': 0.005},
                'tau':   {'prior': {'dist': 'norm', 'loc': 0.054, 'scale': 0.007},
                          'ref': 0.054, 'proposal': 0.003},
                'As':    {'prior': {'min': 1e-9, 'max': 4e-9},   'ref': 2.1e-9, 'proposal': 5e-11},
                'omegamh2': {'derived': 'lambda omch2, ombh2: omch2 + ombh2'},
            },
            'sampler': {'mcmc': mcmc_cfg},
            'output': out_prefix,
        }

        if rank == 0:
            print(f"  Running Cobaya MCMC: {cut['name']} ...")
        cobaya_run(info, resume=True)


# ── Public entry point ───────────────────────────────────────────────────────

def run(args, config):
    """Entry point called by run_pipeline.py for mode 'multipole_cuts'."""
    cfg      = config['multipole_cuts']
    paths_cfg = config['paths']
    sampler  = getattr(args, 'sampler', 'iminuit')

    mask_variants = cfg.get('mask_variants', [''])

    for mask_suffix in mask_variants:
        label = mask_suffix if mask_suffix else '(no extra mask)'
        print(f"\n>> Mask variant: {label}")

        out_dir = os.path.join(paths_cfg['output_folder'],
                               'multipole_cuts', cfg['cls_name'] + mask_suffix)
        os.makedirs(out_dir, exist_ok=True)

        if sampler == 'iminuit':
            _run_iminuit(cfg, paths_cfg, mask_suffix, out_dir)
        elif sampler == 'cobaya':
            _run_cobaya(cfg, paths_cfg, mask_suffix, out_dir)
        else:
            raise ValueError(f"Unknown sampler: {sampler!r}. Use 'iminuit' or 'cobaya'.")

    print("\n>> MODE 2 completed.")