import warnings
warnings.filterwarnings('ignore')
import input_files

import importlib.util
from functools import lru_cache
from functools import partial
from scipy import optimize
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
import truths
from model import model_raw

PID = os.getpid()

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

def inverse_zeng_Mp_Rp_relation(radius, CMF):
    "Rocky planet M-R curves"
    mass = (radius/(1.07 - 0.21*CMF))**3.7
    return mass

def zeng_water(radius):
    "100% water M-R curve"
    log10R = np.array([
        -0.23657201, -0.20065945, -0.16411927, -0.12755235, -0.09140776,
        -0.05601112, -0.02159121,  0.01199311,  0.04414762,  0.07554696,
        0.10585067,  0.1354507 ,  0.16465022,  0.19368103,  0.22245634,
        0.2509077 ,  0.27898212,  0.30599588,  0.33183204,  0.35679046,
        0.38039216,  0.40294883,  0.42488164,  0.44544851,  0.46463856,
        0.4827307 ,  0.49968708,  0.51560895,  0.53058386,  0.54481191,
        0.55822842,  0.57100967,  0.58274497,  0.593729  ,  0.60390183,
        0.61320735,  0.62148786,  0.62900162,  0.63568476,  0.64167237,
        0.64689362,  0.65156874,  0.65561858,  0.65906007,  0.66200188,
        0.66445393,  0.66642437,  0.66801297,  0.66922387
    ])
    log10M = np.array([
        -1.33133458, -1.20943337, -1.08576265, -0.96217525, -0.84013215,
        -0.72033306, -0.60310355, -0.48825029, -0.37613073, -0.26632134,
        -0.1587657 , -0.05227227,  0.05384643,  0.15956719,  0.26505379,
        0.37032801,  0.47494434,  0.57634135,  0.67531998,  0.77151399,
        0.86480763,  0.95607234,  1.04571406,  1.1319393 ,  1.21537315,
        1.29600667,  1.3743817 ,  1.45040309,  1.52491515,  1.59791447,
        1.66950283,  1.73973053,  1.8076703 ,  1.87384353,  1.93876982,
        2.00130093,  2.06182931,  2.1202448 ,  2.17695898,  2.23248787,
        2.28690535,  2.33984878,  2.39199307,  2.44294987,  2.49317912,
        2.54245195,  2.59117595,  2.63938687,  2.68699357
    ])

    log10radius = np.log10(radius)
    if log10radius > log10R[-1] or log10radius < log10R[0]:
        raise ValueError
    
    mass = 10.0**np.interp(np.log10(radius), log10R, log10M)
    return mass

def helper(fcn, radius, radius1):

    radii = np.array([
        1.7, 3.0, 5.0, 10.1
    ])

    points = np.array([
        [np.log10(1.7), np.log10(1.5)],
        [np.log10(4), np.log10(2.5)],
        [np.log10(4), np.log10(4.0)],
        [np.log10(10), np.log10(3)],
    ])

    
    if radius < radius1:
        mass = fcn(radius)
        return mass

    mass1 = fcn(radius1)
    for i in range(len(radii)):
        if radius < radii[i]:
            Rp = radius
        else:
            Rp = radii[i]
        x1, y1 = np.log10(radius1), np.log10(mass1)
        x2, y2 = points[i, :]
        slope = (y2 - y1)/(x2 - x1)
        intercept = y1 - slope*x1
        log10mass = slope*np.log10(Rp) + intercept
        mass = 10.0**log10mass
        
        if radius < radii[i]:
            return mass
        else:
            radius1 = radii[i]
            mass1 = mass
        
    raise ValueError

@lru_cache(maxsize=1)
def find_empirical_and_water_root():
    "Gets intersection between emprical min mass and Zeng water curve."
    def obj(radius):
        return min_mass_empirical(radius) - zeng_water(radius)
    sol = optimize.root_scalar(obj, method='brentq', bracket=[1.0, 2.0], rtol=1.0e-9)
    assert sol.converged
    return sol.root

def min_mass(radius):
    "Minimum possible mass for a given radius"
    radius1 = find_empirical_and_water_root()
    return helper(zeng_water, radius, radius1)

def max_mass(radius):
    "Maximum possible mass for a given radius"

    radius1 = 2.4
    
    if radius < radius1:
        mass = inverse_zeng_Mp_Rp_relation(radius, 1.0)
    else:
        mass1 = inverse_zeng_Mp_Rp_relation(radius1, 1.0)
        x1, y1 = np.log10(radius1), np.log10(mass1)
        x2, y2 = np.log10(7), np.log10(300.0)
        slope = (y2 - y1)/(x2 - x1)
        intercept = y1 - slope*x1
        log10mass = slope*np.log10(radius) + intercept
        mass = 10.0**log10mass

    return mass

def min_mass_empirical(radius):
    "Minimum mass of planets based on observations"
    
    radius1 = 1.25
    
    def fcn(r):
        return inverse_zeng_Mp_Rp_relation(r, 0)

    return helper(fcn, radius, radius1)

def max_mass_notrocky(radius):
    "Maximum mass of a planet that is not rocky (and empirical)"

    if radius > 1.7:
        return max_mass(radius)

    mass = inverse_zeng_Mp_Rp_relation(radius, 0.0)
    return mass

def sample_mass_within_radius_bounds(quantile, log10_Rp):
    radius = 10.0**log10_Rp
    mass1 = min_mass(radius)
    mass2 = max_mass(radius)
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("Mass bounds must be positive")
    log10_mass = quantile_to_uniform(quantile, np.log10(mass1), np.log10(mass2))
    return log10_mass

def _prior_common(cube):
    params = np.zeros_like(cube)
    params[0] = quantile_to_uniform(cube[0], 100.0, 1000.0) # T
    params[1] = quantile_to_uniform(cube[1], -2, 0) # log10_As
    params[2] = quantile_to_uniform(cube[2], -5, 3) # log10_pc
    params[3] = quantile_to_uniform(cube[3], -5, 3) # log10_dpc
    params[4] = quantile_to_uniform(cube[4], -3, 3) # log10_tauc
    params[5] = quantile_to_uniform(cube[5], np.log10(1.0e-14*1.0e-8), np.log10(1.0e-14*1.0e2)) # log10_haze_prod
    params[6] = quantile_to_uniform(cube[6], -3, 0) # log10_fc
    params[7] = quantile_to_uniform(cube[7], np.log10(0.6), 1) # log10_Rp

    params[9] = truncnorm(-5, 5, loc=1.0, scale=0.1).ppf(cube[9]) # a
    params[10] = truncnorm(-5, 5, loc=90.0, scale=9.0).ppf(cube[10]) # phase

    params[12] = quantile_to_uniform(cube[12], 0.0, 1.0) # background H2 fraction
    params[13] = quantile_to_uniform(cube[13], -10, 0) # O2
    params[14] = quantile_to_uniform(cube[14], -10, 0) # H2O
    params[15] = quantile_to_uniform(cube[15], -10, 0) # CO2
    params[16] = quantile_to_uniform(cube[16], -10, 0) # CH4
    params[17] = quantile_to_uniform(cube[17], -10, 0) # O3
    params[18] = quantile_to_uniform(cube[18], -10, 0) # CO
    params[19] = quantile_to_uniform(cube[19], -10, 0) # NH3
    return params

def prior_base(cube):
    params = _prior_common(cube)
    params[8] = sample_mass_within_radius_bounds(cube[8], params[7]) # log10_Mp
    params[11] = realistic_pressure_prior(cube[11], params[8], params[7]) # log10P_surf
    return params

def prior_masserr(cube, mass_mean, mass_error_frac):
    params = _prior_common(cube)

    radius = 10.0**params[7]
    mass1, mass2 = min_mass(radius), max_mass(radius)

    if mass_mean <= 0.0:
        raise ValueError("mass_mean must be positive")
    if mass_error_frac <= 0.0:
        raise ValueError("mass_error_frac must be positive")
    if mass1 <= 0.0:
        raise ValueError("Physical mass lower bound must be positive")

    sigma = mass_mean * mass_error_frac
    a = (mass1 - mass_mean) / sigma
    b = (mass2 - mass_mean) / sigma
    mass = truncnorm(a, b, loc=mass_mean, scale=sigma).ppf(cube[8])
    params[8] = np.log10(mass)  # log10_Mp

    params[11] = realistic_pressure_prior(cube[11], params[8], params[7]) # log10P_surf

    return params

def make_priors(mass_mean, mass_error_frac):
    return partial(prior_masserr, mass_mean=mass_mean, mass_error_frac=mass_error_frac)

@lru_cache(maxsize=1)
def find_max_and_water_root():
    def obj(radius):
        return max_mass(radius) - zeng_water(radius)
    sol = optimize.root_scalar(obj, method='brentq', bracket=[3.0, 4.6], rtol=1.0e-9)
    assert sol.converged
    return sol.root

def must_have_thick_atmosphere(mass, radius):
    "A planet must have a very thick atmosphere if above the 100% water composition curve."

    if radius > find_max_and_water_root():
        return True

    if mass < zeng_water(radius):
        return True
    
    return False

def realistic_pressure_prior(quantile, log10_Mp, log10_Rp):
    mass = 10.0**log10_Mp
    radius = 10.0**log10_Rp

    if must_have_thick_atmosphere(mass, radius):
        return quantile_to_uniform(quantile, 2.9, 3)
    else:
        return quantile_to_uniform(quantile, -5, 3)

def could_be_rocky(mass, radius):
    "Space where a planet can be rocky."

    mass = np.asarray(mass, dtype=float)
    radius = np.asarray(radius, dtype=float)

    upper = inverse_zeng_Mp_Rp_relation(radius, 1.0)
    lower = inverse_zeng_Mp_Rp_relation(radius, 0.0)

    rocky = (radius <= 1.7) & (mass > lower) & (mass < upper)

    return rocky

def empirical_not_rocky(mass, radius):
    """Space where we have found planets, and they canot be a rocky planet.
    The planet could either be mini-Neptune or if the radius is below the 100%
    H2O threshold, then the planet could have a large water envelope.
    """

    mass = np.asarray(mass, dtype=float)
    radius = np.asarray(radius, dtype=float)

    upper = np.array([max_mass_notrocky(a) for a in radius])
    lower = np.array([min_mass_empirical(a) for a in radius])

    res = (mass > lower) & (mass < upper)

    return res

def water_world_possible(mass, radius):
    """Space where water worlds are possible to imagine, but where we have not
    found any planets.
    """

    mass = np.asarray(mass, dtype=float)
    radius = np.asarray(radius, dtype=float)

    radius_max = find_empirical_and_water_root()
    valid = radius < radius_max

    lower = np.full_like(mass, np.nan, dtype=float)
    upper = np.full_like(mass, np.nan, dtype=float)

    idx = valid
    if np.any(idx):
        lower[idx] = np.array([min_mass(a) for a in radius[idx]])
        upper[idx] = np.array([min_mass_empirical(a) for a in radius[idx]])

    res = np.zeros_like(mass, dtype=bool)
    res[idx] = (mass[idx] > lower[idx]) & (mass[idx] < upper[idx])
    return res

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
    bg_h2_fraction = x[12]
    log10_trace = x[13:]

    if log10_pc >= log10P_surf:
        return False
    if np.sum(10.0 ** log10_trace) >= 1.0:
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

def make_loglike(model, opacity, data_dict):

    def loglike(cube):
        data_bins = data_dict['bins']
        y = data_dict['fpfs']
        e = data_dict['err']
        resulty = model(cube, opacity, data_bins)
        if np.any(np.isnan(resulty)):
            return -1.0e100
        loglikelihood = -0.5*np.sum((y - resulty)**2/e**2)
        return loglikelihood
    
    return loglike

def make_loglike_prior(data_dict, param_names, model, model_raw, opacity, prior):

    loglike = make_loglike(model, opacity, data_dict)

    out = {
        'loglike': loglike,
        'data_dict': data_dict,
        'param_names': param_names,
        'model': model,
        'model_raw': model_raw,
        'opacity': opacity,
        'prior': prior,
    }

    return out


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
        "bg_h2_fraction",
        "log10_x_O2",
        "log10_x_H2O",
        "log10_x_CO2",
        "log10_x_CH4",
        "log10_x_O3",
        "log10_x_CO",
        "log10_x_NH3",
    ]

    data_dicts = truths.make_data()

    cases = {}

    # Mini-Neptune no mass constraint
    cases['neptune_clear_nomass'] = make_loglike_prior(
        data_dict=data_dicts['neptune_clear'], 
        param_names=param_names, 
        model=model, 
        model_raw=model_raw, 
        opacity=OPACITY, 
        prior=prior_base
    )

    # Mini-Neptune w/ mass constraint
    truth = data_dicts['neptune_clear']['truth']
    mass = 10.0**truth[8]
    prior = make_priors(
        mass_mean=mass,
        mass_error_frac=0.3,
    )
    cases['neptune_clear_mass'] = make_loglike_prior(
        data_dict=data_dicts['neptune_clear'], 
        param_names=param_names, 
        model=model, 
        model_raw=model_raw, 
        opacity=OPACITY, 
        prior=prior
    )

    # Archean Earth no mass constraint
    cases['archean_clear_nomass'] = make_loglike_prior(
        data_dict=data_dicts['archean_clear'], 
        param_names=param_names, 
        model=model, 
        model_raw=model_raw, 
        opacity=OPACITY, 
        prior=prior_base
    )

    # Archean Earth w/ mass constraint
    truth = data_dicts['archean_clear']['truth']
    mass = 10.0**truth[8]
    prior = make_priors(
        mass_mean=mass,
        mass_error_frac=0.3,
    )
    cases['archean_clear_mass'] = make_loglike_prior(
        data_dict=data_dicts['archean_clear'], 
        param_names=param_names, 
        model=model, 
        model_raw=model_raw, 
        opacity=OPACITY, 
        prior=prior
    )

    return cases

OPACITY = jdi.opannection(
    wave_range=[0.44,1.01],
    filename_db='picasofiles/opacities_photochem_0.1_250.0_R15000_v2.db',
)
RETRIEVAL_CASES = make_cases()
from model import VERBOSE

if __name__ == '__main__':
    patch_pymultinest_analyse()
    nb.set_num_threads(1)
    _ = threadpool_limits(limits=1)

    models_to_run = list(RETRIEVAL_CASES.keys())
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
            LogLikelihood=RETRIEVAL_CASES[model_name]['loglike'], 
            Prior=RETRIEVAL_CASES[model_name]['prior'], 
            n_dims=len(RETRIEVAL_CASES[model_name]['param_names']), 
            outputfiles_basename=outputfiles_basename, 
            verbose=True,
            n_live_points=1000
        )
        # Save pickle
        pickle.dump(results, open(outputfiles_basename+'.pkl','wb'))
