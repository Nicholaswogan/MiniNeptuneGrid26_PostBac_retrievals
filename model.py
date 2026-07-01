import warnings
warnings.filterwarnings('ignore')
import input_files

import utils
import numpy as np
from photochem.utils import stars
from picaso.experimental import interface
from picaso.experimental import utils as eutils
import haze
import os
import time

PID = os.getpid()
VERBOSE = False

HAZE_INTERP = haze.HazeInterpolator('data/haze_optics_grid.h5')

TRACE_SPECIES = ["O2", "H2O", "CO2", "CH4", "O3", "CO", "NH3"]

def untransform(log10_trace, bg_h2_fraction):
    trace = 10.0 ** np.asarray(log10_trace, dtype=float)
    trace_sum = float(np.sum(trace))
    if trace_sum >= 1.0:
        return None

    residual = 1.0 - trace_sum
    bg_h2_fraction = float(np.clip(bg_h2_fraction, 0.0, 1.0))

    mix = {
        "N2": residual * (1.0 - bg_h2_fraction),
        "H2": residual * bg_h2_fraction,
    }
    for sp, val in zip(TRACE_SPECIES, trace):
        mix[sp] = float(val)
    return mix

def model_raw(x, opacity):

    if VERBOSE:
        print(
            f"pid={os.getpid()}: x = {np.array2string(np.asarray(x), precision=17, separator=', ', max_line_width=np.inf)}",
            flush=True,
        )

    def tick(label, t0):
        if VERBOSE:
            print(
                f"pid={PID}: {label}; elapsed={time.time() - t0:.3f}s",
                flush=True
            )

    t0 = time.time()

    T = x[0]
    log10_As = x[1]
    As = 10.0 ** log10_As
    log10_pc = x[2]
    log10_dpc = x[3]
    log10_tauc = x[4]
    # log10_haze_prod = x[5]
    fc = x[5]
    log10_Rp = x[6]
    log10_Mp = x[7]
    a = x[8] # in AU
    phase = x[9] # degrees
    log10P_surf = x[10]
    bg_h2_fraction = x[11]
    log10_trace = x[12:]
    tick("parsed parameters", t0)

    # Get mixing ratios
    species = ['N2', 'H2', 'O2', 'H2O', 'CO2', 'CH4', 'O3', 'CO', 'NH3']
    mix = untransform(log10_trace, bg_h2_fraction)
    if mix is None:
        return np.ones(len(opacity.wno))*np.nan, np.ones(len(opacity.wno))*np.nan, np.ones(len(opacity.wno))*np.nan
    P_surf = 10.0**log10P_surf

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

    # # Get haze df
    # m = haze.McKayTitanHazeModel(sweep_clear_below_pressure=1.0e7)
    # tick("initialized haze model", t0)

    # solution = m.solve_from_atmosphere(
    #     atm,
    #     column_production=10.0**log10_haze_prod,
    #     peak_pressure=1.0e-6,
    #     width_pressure=1.0e-6 * np.exp(-1.124),
    #     planet_radius=10.0**log10_Rp,
    #     planet_mass=10.0**log10_Mp,
    #     reference_pressure=p_reference,
    # )
    # tick("solved haze model", t0)

    # haze_df = haze.make_picaso_haze_clouddf_from_solution(
    #     solution, 
    #     optics_function=HAZE_INTERP
    # )
    # tick("built haze dataframe", t0)

    # Initialize class
    planet = utils.initialize_model(
        opacity,
        atm,
        phase=phase*np.pi/180.0,
        num_gangle=4, 
        num_tangle=4,
        surface_albedo=As,
        semi_major=a,
        stellar_radius=1.0,
        planet_radius=10.0**log10_Rp,
        planet_mass=10.0**log10_Mp,
        p_reference=p_reference,
        cloud_frac=fc,
        cloud_df=cloud_df
    )
    tick("initialized PICASO model", t0)

    # Compute spectrum
    # df = utils.spectrum(planet, opacity, cloud_df, haze_df, water_cloud_frac=10.0**log10_fc)
    df = planet.spectrum(opacity, calculation='reflected')
    tick("computed spectrum", t0)

    tick("finished model_raw", t0)

    return get_result(df, opacity)

def model_hazy_raw(x, opacity):

    if VERBOSE:
        print(
            f"pid={os.getpid()}: x = {np.array2string(np.asarray(x), precision=17, separator=', ', max_line_width=np.inf)}",
            flush=True,
        )

    def tick(label, t0):
        if VERBOSE:
            print(
                f"pid={PID}: {label}; elapsed={time.time() - t0:.3f}s",
                flush=True
            )

    t0 = time.time()

    T = x[0]
    log10_As = x[1]
    As = 10.0 ** log10_As
    log10_pc = x[2]
    log10_dpc = x[3]
    log10_tauc = x[4]
    log10_haze_prod = x[5]
    fc = x[6]
    log10_Rp = x[7]
    log10_Mp = x[8]
    a = x[9] # in AU
    phase = x[10] # degrees
    log10P_surf = x[11]
    bg_h2_fraction = x[12]
    log10_trace = x[13:]
    tick("parsed parameters", t0)

    # Get mixing ratios
    species = ['N2', 'H2', 'O2', 'H2O', 'CO2', 'CH4', 'O3', 'CO', 'NH3']
    mix = untransform(log10_trace, bg_h2_fraction)
    if mix is None:
        return np.ones(len(opacity.wno))*np.nan, np.ones(len(opacity.wno))*np.nan, np.ones(len(opacity.wno))*np.nan
    P_surf = 10.0**log10P_surf

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
        optics_function=HAZE_INTERP
    )
    tick("built haze dataframe", t0)

    # Initialize class
    planet = utils.initialize_model(
        opacity,
        atm,
        phase=phase*np.pi/180.0,
        num_gangle=4, 
        num_tangle=4,
        surface_albedo=As,
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
    df = utils.spectrum(planet, opacity, cloud_df, haze_df, water_cloud_frac=fc)
    # df = planet.spectrum(opacity, calculation='reflected')
    tick("computed spectrum", t0)

    tick("finished model_raw", t0)

    return get_result(df, opacity)

def get_result(df, opacity):

    if isinstance(opacity, interface.ExperimentalRT):
        reflected_result = opacity.rad.reflected_result
        bin_edges = reflected_result.bin_edges
        albedo = df['albedo'][::-1].copy()
        fpfs = df['fpfs_reflected'][::-1].copy()
    else:
        wv = 1e4/df['wavenumber'][::-1].copy()
        albedo = df['albedo'][::-1].copy()
        fpfs = df['fpfs_reflected'][::-1].copy()
        wavl = stars.make_bins(wv)
        bin_edges = eutils.bin_edges_from_wavelength_edges(wavl)

    return bin_edges.copy(), fpfs.copy(), albedo.copy()
