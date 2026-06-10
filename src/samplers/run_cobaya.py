#%% Imports
import os
import numpy as np
from cobaya.run import run
from getdist import loadMCSamples
from src.utils.data_utils import slice_cls

#%% Functions
def build_cobaya_info(cut_cl, cut_std, cut_edges, ref_dict_PR3, out_prefix, config, a_ps_dynamic):
    """Construye el diccionario de información para Cobaya."""
    
    # Aquí puedes llamar a una función de src/utils/iminuit_utils.py para tu likelihood externa
    # Pero para no marear, asumimos que tienes `get_likelihood` importada.
    from src.samplers.run_iminuit import get_cobaya_likelihood 

    info = {
        "params": {
            "ombh2": {"prior": {"min": 0.01, "max": 0.05}, "ref": ref_dict_PR3['ombh2'], "proposal": 0.0001, "latex": r"\omega_b"},
            "omch2": {"prior": {"min": 0.09, "max": 0.40}, "ref": ref_dict_PR3['omch2'], "proposal": 0.001, "latex": r"\omega_c"},
            "ns":    {"prior": {"min": 0.8, "max": 0.999}, "ref": ref_dict_PR3['ns'], "proposal": 0.005, "latex": r"n_s"},
            "logA":  {"prior": {"min": 2.5, "max": 3.5}, "ref": np.log(1e10 * ref_dict_PR3['As']), "proposal": 0.01, "drop": True, "latex": r"\ln(10^{10} A_s)"},
            "As":    {"value": "lambda logA: np.exp(logA) / 1e10", "latex": r"A_s"},
            "H0":    {"prior": {"min": 30.0, "max": 110.0}, "ref": ref_dict_PR3['H0'], "proposal": 0.5, "latex": "H_0"},
            "tau": {"prior": {"min": 0.01, "max": 0.15}, "ref": ref_dict_PR3['tau'], "proposal": 0.005, "latex": r"\tau"},
            "a_ps":  a_ps_dynamic,
            "omegamh2": {"derived": "lambda ombh2, omch2: ombh2 + omch2", "latex": r"\omega_m"},
            "As_e2tau": {"derived": "lambda As, tau: As * np.exp(-2 * tau) * 1e9", "latex": r"10^9 A_s e^{-2\tau}"}
        },
        "likelihood": {
            "my_cmb_like": {"external": get_cobaya_likelihood(cut_cl, cut_std, cut_edges), "requires": {"Cl": {"tt": int(cut_edges[-1])}}}
        },
        "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 0, "num_massive_neutrinos": 1}}},
        "sampler": {"mcmc": {"Rminus1_stop": config['mcmc_setup']['Rminus1_stop'], "max_tries": config['mcmc_setup']['max_tries']}},
        "output": out_prefix,
        "resume": True
    }
    return info

def run_mcmc(cut_info, data_cl, data_std, ell_edges, ref_dict_PR3, out_prefix, config, rank=0):
    """Ejecuta el pipeline de Cobaya y retorna las estadísticas (mean, std)."""
    # 1. Cortar datos
    cut_cl, cut_std, cut_edges = slice_cls(cut_info['ell_cut'], data_cl, data_std, ell_edges, cut_info['low_cut'])
    if len(cut_cl) < 5: 
        return None

    # 2. Configurar a_ps dinámico (tu lógica original)
    a_ps_setup = {"prior": {"min": 0.0, "max": 300.0}, "ref": ref_dict_PR3['a_ps'], "proposal": 2.0}
    if cut_info['ell_cut'] is not None and not cut_info['low_cut'] and cut_info['ell_cut'] <= 1250:
        a_ps_setup = ref_dict_PR3['a_ps']

    info = build_cobaya_info(cut_cl, cut_std, cut_edges, ref_dict_PR3, out_prefix, config, a_ps_setup)

    # 3. Correr MCMC si no existe
    if not os.path.exists(out_prefix + ".1.txt"):
        if rank == 0: print(f"-> Corriendo MCMC para {cut_info['name']}...")
        run(info)
    else:
        if rank == 0: print(f"-> Cadena encontrada para {cut_info['name']}. Omitiendo ejecución...")

    # 4. Extraer Stats
    stats = {}
    if rank == 0:
        samples = loadMCSamples(out_prefix, settings={'ignore_rows': 0.3})
        params_names = ['H0', 'ombh2', 'omch2', 'ns', 'As_e2tau', 'tau'] # Los que te interesan
        for param in params_names:
            try:
                mean = samples.getMeans()[samples.index[param]]
                std = np.sqrt(samples.getVars()[samples.index[param]])
                stats[param] = {'mean': mean, 'std': std}
            except KeyError:
                continue
    return stats