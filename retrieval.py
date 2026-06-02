import warnings
warnings.filterwarnings('ignore')
import input_files

import importlib.util
from pathlib import Path
import utils
import numpy as np
from photochem.utils import stars
from scipy.stats import truncnorm
import haze
import pickle
from pymultinest.solve import solve
import os
from picaso import justdoit as jdi
import numba as nb
from threadpoolctl import threadpool_limits
import time
import pymultinest.analyse as pymultinest_analyse

PID = os.getpid()
LOG_LIKE_COUNT = 0


def patch_pymultinest_analyse():
    """Force pymultinest to use the local Fortran-exponent parser."""
    local_analyse_path = Path(__file__).with_name("analyse.py")
    if not local_analyse_path.exists():
        return

    spec = importlib.util.spec_from_file_location(
        "_codex_local_pymultinest_analyse",
        local_analyse_path,
    )
    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pymultinest_analyse.loadtxt2d = module.loadtxt2d


def quantile_to_uniform(quantile, lower_bound, upper_bound):
    return quantile*(upper_bound - lower_bound) + lower_bound

def untransform(log10u_i):
    u_i = 10.0**log10u_i
    x_i = u_i/np.sum(u_i)
    return x_i

def model_raw(x, opacity, R=None):

    if VERBOSE:
        print(
            f"pid={os.getpid()}: x = {np.array2string(np.asarray(x), precision=17, separator=', ', max_line_width=np.inf)}",
            flush=True,
        )

    def tick(label, t0):
        if VERBOSE:
            print(
                f"pid={PID}: {label}; elapsed={time.time() - t0:.2f}s",
                flush=True
            )

    t0 = time.time()

    T = x[0]
    log10_As = x[1]
    log10_pc = x[2]
    log10_dpc = x[3]
    log10_tauc = x[4]
    log10_haze_prod = x[5]
    log10_fc = x[6]
    log10_Rp = x[7]
    log10_Mp = x[8]
    a = x[9] # in AU
    phase = x[10] # degrees
    log10P_surf = x[11]
    log10u_i = x[12:]
    tick("parsed parameters", t0)

    # Get mixing ratios
    species = ['N2', 'O2', 'H2O', 'CO2', 'CH4', 'O3', 'H2', 'CO']
    mix_i = untransform(log10u_i)
    P_surf = 10.0**log10P_surf

    mix = {}
    for i,sp in enumerate(species):
        mix[sp] = mix_i[i]
    tick("computed mixing ratios", t0)

    # Build atmosphere
    atm = utils.build_atmosphere(mix, T, np.log10(P_surf), log10_P_top=-8.0, nlevels=50)
    tick("built atmosphere", t0)
    p_reference = np.clip(min(1.0, P_surf), float(np.min(atm["pressure"])), float(np.max(atm["pressure"])))

    # Get cloud df
    cloud_df = utils.build_cloud_df(
        atm,
        cloud_scheme="rfast",
        cloud_top_pressure=10.0**log10_pc,
        cloud_thickness=10.0**log10_dpc,
        cloud_opd=10.0**log10_tauc,
        cloud_w0=0.99,
        cloud_g0=0.85,
    )
    tick("built cloud dataframe", t0)

    # Get haze df
    m = haze.McKayTitanHazeModel(sweep_clear_below_pressure=1.0e7)
    tick("initialized haze model", t0)

    solution = m.solve_from_atmosphere(
        atm,
        column_production=10.0**log10_haze_prod,
        peak_pressure=1.0e-6,
        width_pressure=1.0e-6 * np.exp(-1.124),
        planet_radius=10.0**log10_Rp,
        planet_mass=10.0**log10_Mp,
        reference_pressure=p_reference,
    )
    tick("solved haze model", t0)

    haze_df = haze.make_picaso_haze_clouddf_from_solution(
        solution, 
        refractive_index_file='data/khare_tholins.refrind'
    )
    tick("built haze dataframe", t0)

    # Initialize class
    planet = utils.initialize_model(
        opacity,
        atm,
        phase=phase*np.pi/180.0,
        num_gangle=4, 
        num_tangle=4,
        surface_albedo=10.0**log10_As,
        stellar_teff=5780.0,
        stellar_metallicity=0.0,
        stellar_logg=4.0,
        semi_major=a,
        stellar_radius=1.0,
        planet_radius=10.0**log10_Rp,
        planet_mass=10.0**log10_Mp,
        p_reference=p_reference,
        cloud_frac=None,
        cloud_df=None
    )
    tick("initialized PICASO model", t0)

    # Compute spectrum
    df = utils.spectrum(planet, opacity, cloud_df, haze_df, water_cloud_frac=10.0**log10_fc)
    tick("computed spectrum", t0)

    # Unpack result
    wv = 1e4/df['wavenumber'][::-1].copy()
    albedo = df['albedo'][::-1].copy()
    fpfs = df['fpfs_reflected'][::-1].copy()
    tick("unpacked spectrum", t0)

    if R is not None:
        wavl = stars.make_bins(wv)
        wavl_new = stars.grid_at_resolution(wavl[0], wavl[-1], R)
        wv_new = (wavl_new[1:] + wavl_new[:-1])/2
        albedo_new = stars.rebin(wavl, albedo, wavl_new)
        fpfs_new = stars.rebin(wavl, fpfs, wavl_new)

        wv = wv_new
        albedo = albedo_new
        fpfs = fpfs_new
        tick(f"rebinned to R={R}", t0)

    tick("finished model_raw", t0)

    return wv, albedo, fpfs

def prior(cube):
    params = np.zeros_like(cube)
    params[0] = quantile_to_uniform(cube[0], 100.0, 1000.0) # T
    params[1] = quantile_to_uniform(cube[1], -2, 0) # log10_As
    params[2] = quantile_to_uniform(cube[2], -5, 3) # log10_pc
    params[3] = quantile_to_uniform(cube[3], -5, 3) # log10_dpc
    params[4] = quantile_to_uniform(cube[4], -3, 3) # log10_tauc
    params[5] = quantile_to_uniform(cube[5], np.log10(1.0e-14*1.0e-8), np.log10(1.0e-14*1.0e2)) # log10_haze_prod
    params[6] = quantile_to_uniform(cube[6], -3, 0) # log10_fc
    params[7] = quantile_to_uniform(cube[7], -1, 1) # log10_Rp
    params[8] = quantile_to_uniform(cube[8], -1, 2) # log10_Mp
    params[9] = truncnorm(-5, 5, loc=1.0, scale=0.1).ppf(cube[9]) # a
    params[10] = truncnorm(-5, 5, loc=90.0, scale=9.0).ppf(cube[10]) # phase
    params[11] = quantile_to_uniform(cube[11], -5, 3) # log10P_surf
    params[12] = quantile_to_uniform(cube[12], -13, 0) # N2
    params[13] = quantile_to_uniform(cube[13], -13, 0) # O2
    params[14] = quantile_to_uniform(cube[14], -13, 0) # H2O
    params[15] = quantile_to_uniform(cube[15], -13, 0) # CO2
    params[16] = quantile_to_uniform(cube[16], -13, 0) # CH4
    params[17] = quantile_to_uniform(cube[17], -13, 0) # O3
    params[18] = quantile_to_uniform(cube[18], -13, 0) # H2
    params[19] = quantile_to_uniform(cube[19], -13, 0) # CO
    return params  

def implicit_priors(x):
    T = x[0]
    log10_As = x[1]
    log10_pc = x[2]
    log10_dpc = x[3]
    log10_tauc = x[4]
    log10_haze_prod = x[5]
    log10_fc = x[6]
    log10_Rp = x[7]
    log10_Mp = x[8]
    a = x[9]
    phase = x[10]
    log10P_surf = x[11]
    log10u_i = x[12:]

    if log10_pc >= log10P_surf:
        return False
    
    return True

def model(x, opacity, wv_bins):

    within_implicit_priors = implicit_priors(x)
    if not within_implicit_priors:
        return np.zeros(len(wv_bins))*np.nan
    
    wv, albedo, fpfs1 = model_raw(x, opacity)
    wavl = stars.make_bins(wv)

    fpfs = np.empty(len(wv_bins))
    for i,b in enumerate(wv_bins):
        fpfs[i] = stars.rebin(wavl.copy(), fpfs1.copy(), b.copy())
    
    return fpfs

def loglike(cube, data_name):
    global LOG_LIKE_COUNT
    LOG_LIKE_COUNT += 1
    eval_id = LOG_LIKE_COUNT
    t0 = time.time()

    if VERBOSE:
        print(f"pid={PID}: entered loglike #{eval_id} ({data_name})", flush=True)

    data_dict = DATA_DICTS[data_name]
    data_bins = data_dict['bins']
    y = data_dict['fpfs']
    e = data_dict['err']

    if VERBOSE:
        print(f"pid={PID}: before model() #{eval_id}", flush=True)

    t_model0 = time.time()
    resulty = model(cube, OPACITY, data_bins)
    t_model1 = time.time()

    model_time = t_model1 - t_model0
    total_time = t_model1 - t0

    if VERBOSE:
        print(
            f"pid={PID}: after model() #{eval_id}; "
            f"model_time={model_time:.2f}s total_time={total_time:.2f}s",
            flush=True
        )

    if np.any(np.isnan(resulty)):
        if VERBOSE:
            print(
                f"pid={PID}: returning -1e100 (nan) #{eval_id}; "
                f"model_time={model_time:.2f}s total_time={total_time:.2f}s",
                flush=True
            )
        return -1.0e100

    loglikelihood = -0.5*np.sum((y - resulty)**2/e**2)

    if VERBOSE:
        print(
            f"pid={PID}: returning loglike #{eval_id} = {loglikelihood:.6e}; "
            f"model_time={model_time:.2f}s total_time={total_time:.2f}s",
            flush=True
        )

    return loglikelihood

def loglike_clear(cube):
    return loglike(cube, 'clear')

def loglike_hazy(cube):
    return loglike(cube, 'hazy')

def make_cases():

    param_names = [
        "T",
        "log10_As",
        "log10_pc",
        "log10_dpc",
        "log10_tauc",
        "log10_haze_prod",
        "log10_fc",
        "log10_Rp",
        "log10_Mp",
        "a",
        "phase",
        "log10P_surf",
        "log10u_N2",
        "log10u_O2",
        "log10u_H2O",
        "log10u_CO2",
        "log10u_CH4",
        "log10u_O3",
        "log10u_H2",
        "log10u_CO",
    ]
    with open('data/neptune_20.pkl','rb') as f:
        data = pickle.load(f)

    retrieval_names = ['clear', 'hazy']
    data_dicts = {
        'clear': data['clear'],
        'hazy': data['hazy'],
    }
    param_names_out = {
        'clear': param_names,
        'hazy': param_names
    }

    return retrieval_names, data_dicts, param_names_out

OPACITY = jdi.opannection(
    wave_range=[0.4,1.85],
    filename_db='picasofiles/opacities_photochem_0.1_250.0_R15000_v2.db',
)
RETRIEVAL_NAMES, DATA_DICTS, PARAM_NAMES = make_cases()
LOGLIKES = {
    'clear': loglike_clear,
    'hazy': loglike_hazy,
}
PRIORS = {
    'clear': prior,
    'hazy': prior
}
VERBOSE = False

if __name__ == '__main__':
    patch_pymultinest_analyse()
    nb.set_num_threads(1)
    _ = threadpool_limits(limits=1)

    models_to_run = RETRIEVAL_NAMES
    for model_name in models_to_run:
        # Setup directories
        outputfiles_basename = f'pymultinest/{model_name}/{model_name}'
        try:
            os.mkdir(f'pymultinest/{model_name}')
        except FileExistsError:
            pass

        # Do nested sampling
        print(f"pid={PID}: starting solve for {model_name}", flush=True)
        results = solve(
            LogLikelihood=LOGLIKES[model_name], 
            Prior=PRIORS[model_name], 
            n_dims=len(PARAM_NAMES[model_name]), 
            outputfiles_basename=outputfiles_basename, 
            verbose=True,
            n_live_points=1000
        )
        # Save pickle
        with open(outputfiles_basename + ".pkl", "wb") as f:
            pickle.dump(results, f)
