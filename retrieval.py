import warnings
warnings.filterwarnings('ignore')
import input_files

import importlib.util
from functools import lru_cache
from functools import partial
from scipy import optimize
from pathlib import Path
import utils
from utils import inverse_zeng_Mp_Rp_relation, zeng_water
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
from model import model_raw, model_hazy_raw
from picaso.experimental import utils as eutils
from picaso.experimental import interface

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

@lru_cache(maxsize=1)
def _mass_radius_support_grid():
    """Precompute monotonic mass-radius envelopes for fast inversion."""
    radii = np.logspace(np.log10(0.6), 1.0, 2048)
    min_masses = np.array([min_mass(r) for r in radii], dtype=float)
    max_masses = np.array([max_mass(r) for r in radii], dtype=float)
    min_masses = np.maximum.accumulate(min_masses)
    max_masses = np.maximum.accumulate(max_masses)
    return radii, min_masses, max_masses

def mass_bounds_for_radius_support():
    """Mass interval supported by the current radius prior support."""
    radii, min_masses, max_masses = _mass_radius_support_grid()
    return float(min_masses[0]), float(max_masses[-1])

def radius_bounds_for_mass(mass):
    """Return the allowed radius interval for a given mass."""
    radii, min_masses, max_masses = _mass_radius_support_grid()

    mass = float(mass)
    mass_min = float(min_masses[0])
    mass_max = float(max_masses[-1])
    if mass < mass_min or mass > mass_max:
        raise ValueError(
            f"Mass {mass} lies outside the supported mass-radius envelope "
            f"[{mass_min}, {mass_max}]"
        )

    r_min = float(np.interp(mass, max_masses, radii, left=radii[0], right=radii[-1]))
    r_max = float(np.interp(mass, min_masses, radii, left=radii[0], right=radii[-1]))

    r_min = max(r_min, float(radii[0]))
    r_max = min(r_max, float(radii[-1]))
    if r_min > r_max:
        raise ValueError(
            f"No physical radius interval exists for mass {mass}; "
            f"got r_min={r_min}, r_max={r_max}"
        )
    return r_min, r_max

def sample_radius_within_mass_bounds(quantile, mass):
    """Sample log10(radius) uniformly within the allowed interval for mass."""
    r_min, r_max = radius_bounds_for_mass(mass)
    if r_min <= 0.0 or r_max <= 0.0:
        raise ValueError("Radius bounds must be positive")
    return quantile_to_uniform(quantile, np.log10(r_min), np.log10(r_max))

def sample_radius_within_mass_bounds_linear(quantile, mass):
    """Sample radius uniformly within the allowed interval for mass."""
    r_min, r_max = radius_bounds_for_mass(mass)
    if r_min <= 0.0 or r_max <= 0.0:
        raise ValueError("Radius bounds must be positive")
    return quantile_to_uniform(quantile, r_min, r_max)

def _prior_common(cube):
    params = np.zeros_like(cube)
    params[0] = quantile_to_uniform(cube[0], 100.0, 1000.0) # T
    params[1] = quantile_to_uniform(cube[1], -2, 0) # log10_As
    params[2] = quantile_to_uniform(cube[2], -5, 3) # log10_pc
    params[3] = quantile_to_uniform(cube[3], -5, 3) # log10_dpc
    params[4] = quantile_to_uniform(cube[4], -3, 3) # log10_tauc
    params[5] = quantile_to_uniform(cube[5], 0, 1) # fc
    params[6] = quantile_to_uniform(cube[6], np.log10(0.6), 1) # log10_Rp

    params[8] = truncnorm(-5, 5, loc=1.0, scale=0.1).ppf(cube[8]) # a
    params[9] = truncnorm(-5, 5, loc=90.0, scale=9.0).ppf(cube[9]) # phase

    params[11] = quantile_to_uniform(cube[11], 0.0, 1.0) # background H2 fraction
    params[12] = quantile_to_uniform(cube[12], -10, 0) # O2
    params[13] = quantile_to_uniform(cube[13], -10, 0) # H2O
    params[14] = quantile_to_uniform(cube[14], -10, 0) # CO2
    params[15] = quantile_to_uniform(cube[15], -10, 0) # CH4
    params[16] = quantile_to_uniform(cube[16], -10, 0) # O3
    params[17] = quantile_to_uniform(cube[17], -10, 0) # CO
    params[18] = quantile_to_uniform(cube[18], -10, 0) # NH3
    return params

def _prior_common_hazy(cube):
    params = np.zeros_like(cube)
    params[0] = quantile_to_uniform(cube[0], 100.0, 1000.0) # T
    params[1] = quantile_to_uniform(cube[1], -2, 0) # log10_As
    params[2] = quantile_to_uniform(cube[2], -5, 3) # log10_pc
    params[3] = quantile_to_uniform(cube[3], -5, 3) # log10_dpc
    params[4] = quantile_to_uniform(cube[4], -3, 3) # log10_tauc
    params[5] = quantile_to_uniform(cube[5], -19, -12) # log10_haze_prod
    params[6] = quantile_to_uniform(cube[6], 0, 1) # fc
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
    params[7] = sample_mass_within_radius_bounds(cube[7], params[6]) # log10_Mp
    params[10] = realistic_pressure_prior(cube[10], params[7], params[6]) # log10P_surf
    return params

def prior_base_linearRp(cube):
    params = _prior_common(cube)
    radius = quantile_to_uniform(cube[6], 0.6, 10.0)
    params[6] = np.log10(radius) # log10_Rp
    params[7] = sample_mass_within_radius_bounds(cube[7], params[6]) # log10_Mp
    params[10] = realistic_pressure_prior(cube[10], params[7], params[6]) # log10P_surf
    return params

def prior_masserr_linearRp(cube, mass_mean, mass_error_frac):
    params = _prior_common(cube)

    if mass_mean <= 0.0:
        raise ValueError("mass_mean must be positive")
    if mass_error_frac <= 0.0:
        raise ValueError("mass_error_frac must be positive")

    sigma = mass_mean * mass_error_frac
    mass1, mass2 = mass_bounds_for_radius_support()
    a = (mass1 - mass_mean) / sigma
    b = (mass2 - mass_mean) / sigma
    mass = truncnorm(a, b, loc=mass_mean, scale=sigma).ppf(cube[7])
    params[7] = np.log10(mass)  # log10_Mp

    radius = sample_radius_within_mass_bounds_linear(cube[6], mass)
    params[6] = np.log10(radius)  # log10_Rp

    params[10] = realistic_pressure_prior(cube[10], params[7], params[6]) # log10P_surf

    return params

def prior_base_hazy(cube):
    params = _prior_common_hazy(cube)
    params[8] = sample_mass_within_radius_bounds(cube[8], params[7]) # log10_Mp
    params[11] = realistic_pressure_prior(cube[11], params[8], params[7]) # log10P_surf
    return params

def prior_masserr(cube, mass_mean, mass_error_frac):
    params = _prior_common(cube)

    if mass_mean <= 0.0:
        raise ValueError("mass_mean must be positive")
    if mass_error_frac <= 0.0:
        raise ValueError("mass_error_frac must be positive")

    sigma = mass_mean * mass_error_frac
    mass1, mass2 = mass_bounds_for_radius_support()
    a = (mass1 - mass_mean) / sigma
    b = (mass2 - mass_mean) / sigma
    mass = truncnorm(a, b, loc=mass_mean, scale=sigma).ppf(cube[7])
    params[7] = np.log10(mass)  # log10_Mp

    params[6] = sample_radius_within_mass_bounds(cube[6], mass)  # log10_Rp

    params[10] = realistic_pressure_prior(cube[10], params[7], params[6]) # log10P_surf

    return params

def prior_masserr_hazy(cube, mass_mean, mass_error_frac):
    params = _prior_common_hazy(cube)

    if mass_mean <= 0.0:
        raise ValueError("mass_mean must be positive")
    if mass_error_frac <= 0.0:
        raise ValueError("mass_error_frac must be positive")

    sigma = mass_mean * mass_error_frac
    mass1, mass2 = mass_bounds_for_radius_support()
    a = (mass1 - mass_mean) / sigma
    b = (mass2 - mass_mean) / sigma
    mass = truncnorm(a, b, loc=mass_mean, scale=sigma).ppf(cube[8])
    params[8] = np.log10(mass)  # log10_Mp

    params[7] = sample_radius_within_mass_bounds(cube[7], mass)  # log10_Rp

    params[11] = realistic_pressure_prior(cube[11], params[8], params[7]) # log10P_surf

    return params

def make_priors(mass_mean, mass_error_frac):
    return partial(prior_masserr, mass_mean=mass_mean, mass_error_frac=mass_error_frac)

def make_priors_hazy(mass_mean, mass_error_frac):
    return partial(prior_masserr_hazy, mass_mean=mass_mean, mass_error_frac=mass_error_frac)

def make_priors_linearRp(mass_mean, mass_error_frac):
    return partial(prior_masserr_linearRp, mass_mean=mass_mean, mass_error_frac=mass_error_frac)

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
        return quantile_to_uniform(quantile, 1, 3)
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

def implicit_priors(x, hazy=False):
    T = x[0]
    log10_As = x[1]
    As = 10.0 ** log10_As
    log10_pc = x[2]
    log10_dpc = x[3]
    log10_tauc = x[4]
    if hazy:
        log10_haze_prod = x[5]
        fc = x[6]
        log10_Rp = x[7]
        log10_Mp = x[8]
        a = x[9]
        phase = x[10]
        log10P_surf = x[11]
        bg_h2_fraction = x[12]
        log10_trace = x[13:]
    else:
        fc = x[5]
        log10_Rp = x[6]
        log10_Mp = x[7]
        a = x[8]
        phase = x[9]
        log10P_surf = x[10]
        bg_h2_fraction = x[11]
        log10_trace = x[12:]

    if log10_pc >= log10P_surf:
        return False
    if np.sum(10.0 ** log10_trace) >= 1.0:
        return False
    
    return True

def model(x, opacity, wv_bins):

    within_implicit_priors = implicit_priors(x, hazy=False)
    if not within_implicit_priors:
        return np.zeros(len(wv_bins))*np.nan
    
    bin_edges, fpfs1, _ = model_raw(x, opacity)
    fpfs = eutils.rebin_edges(bin_edges, fpfs1, wv_bins)
    
    return fpfs

def model_hazy(x, opacity, wv_bins):

    within_implicit_priors = implicit_priors(x, hazy=True)
    if not within_implicit_priors:
        return np.zeros(len(wv_bins))*np.nan
    
    bin_edges, fpfs1, _ = model_hazy_raw(x, opacity)
    fpfs = eutils.rebin_edges(bin_edges, fpfs1, wv_bins)
    
    return fpfs

def make_loglike(model, opacity, data_dict):

    def loglike(cube):
        data_bins = data_dict['bins']
        y = data_dict['fpfs']
        e = data_dict['err']
        try:
            resulty = model(cube, opacity, data_bins)
        except ValueError as err:
            # If error then we return tiny
            return -1.0e100
        if np.any(~np.isfinite(resulty)):
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
        "fc",
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

    param_names_hazy = [
        "T",
        "log10_As",
        "log10_pc",
        "log10_dpc",
        "log10_tauc",
        "log10_haze_prod",
        "fc",
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

    opacities = {
        'gap': OPACITY_GAP,
        'nogap': OPACITY_NOGAP
    }
    planet_types = ['neptune', 'archean', 'superarchean']
    data_cases = ['gap', 'nogap']
    mass_precisions = [None, 0.5, 0.3, 0.1]
    for i,planet_type in enumerate(planet_types):
        for j,data_case in enumerate(data_cases):
            data_dict = data_dicts[f'{planet_type}_{data_case}']
            for k,mass_precision in enumerate(mass_precisions):
                
                if mass_precision is None:
                    prior = prior_base
                else:
                    truth = data_dict['truth']
                    mass = 10.0**truth[7]
                    prior = make_priors(
                        mass_mean=mass,
                        mass_error_frac=mass_precision,
                    )

                label = f'{planet_type}_{data_case}_{mass_precision}'
                cases[label] = make_loglike_prior(
                    data_dict=data_dict, 
                    param_names=param_names, 
                    model=model, 
                    model_raw=model_raw, 
                    opacity=opacities[data_case], 
                    prior=prior
                )

    for i,planet_type in enumerate(planet_types):
        for j,data_case in enumerate(data_cases):
            data_dict = data_dicts[f'{planet_type}_{data_case}']
            for k,mass_precision in enumerate(mass_precisions):
                
                if mass_precision is None:
                    prior = prior_base_linearRp
                else:
                    truth = data_dict['truth']
                    mass = 10.0**truth[7]
                    prior = make_priors_linearRp(
                        mass_mean=mass,
                        mass_error_frac=mass_precision,
                    )

                label = f'{planet_type}_{data_case}_{mass_precision}_linearRp'
                cases[label] = make_loglike_prior(
                    data_dict=data_dict,
                    param_names=param_names,
                    model=model,
                    model_raw=model_raw,
                    opacity=opacities[data_case],
                    prior=prior,
                )

    planet_types = ['neptunehazy', 'archeanhazy', 'superarcheanhazy']
    data_case = 'nogap'
    mass_precision = None
    for i,planet_type in enumerate(planet_types):

        data_dict = data_dicts[f'{planet_type}_{data_case}']
        prior = prior_base

        label = f'{planet_type}_{data_case}_{mass_precision}'
        cases[label] = make_loglike_prior(
            data_dict=data_dict, 
            param_names=param_names_hazy, 
            model=model_hazy, 
            model_raw=model_hazy_raw, 
            opacity=opacities[data_case], 
            prior=prior_base_hazy
        )

    return cases

OPACITY_GAP = interface.opannection(
    filename_db='picasofiles/opacities_ck_gap.h5',
)
OPACITY_NOGAP = interface.opannection(
    filename_db='picasofiles/opacities_ck_nogap.h5',
)
RETRIEVAL_CASES = make_cases()

if __name__ == '__main__':
    patch_pymultinest_analyse()
    nb.set_num_threads(1)
    _ = threadpool_limits(limits=1)

    # models_to_run = list(RETRIEVAL_CASES.keys())
    models_to_run = sorted([
        name for name in RETRIEVAL_CASES
        if name.endswith('_linearRp') and 'hazy' not in name
    ])
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

# 3 planet types = [Archean, Super Archean, Mini Neptune]
# 2 data types = [Small, Big]
# 4 mass constraints = [No mass, 50%, 30%, 10%]
# 24 total cases
