import warnings
warnings.filterwarnings('ignore')
import input_files

import utils
import numpy as np
from photochem.utils import stars
from scipy.stats import truncnorm
import haze

def quantile_to_uniform(quantile, lower_bound, upper_bound):
    return quantile*(upper_bound - lower_bound) + lower_bound

def untransform(log10u_i):
    u_i = 10.0**log10u_i
    x_i = u_i/np.sum(u_i)
    return x_i

def model_raw(x, opacity, R=None):
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

    # Get mixing ratios
    species = ['N2', 'O2', 'H2O', 'CO2', 'CH4', 'O3', 'H2', 'CO']
    mix_i = untransform(log10u_i)
    P_surf = 10.0**log10P_surf
    mix = {}
    for i,sp in enumerate(species):
        mix[sp] = mix_i[i]

    # Build atmosphere
    atm = utils.build_atmosphere(mix, T, np.log10(P_surf), log10_P_top=-7.0, nlevels=90)

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

    # Get haze df
    m = haze.McKayTitanHazeModel(sweep_clear_below_pressure=1.0e7)
    solution = m.solve_from_atmosphere(
        atm,
        column_production=10.0**log10_haze_prod,
        peak_pressure=1.0e-6,
        width_pressure=1.0e-6 * np.exp(-1.124),
        planet_radius=10.0**log10_Rp,
        planet_mass=10.0**log10_Mp,
        reference_pressure=np.minimum(1.0, P_surf),
    )
    haze_df = haze.make_picaso_haze_clouddf_from_solution(
        solution, 
        refractive_index_file='data/khare_tholins.refrind'
    )

    # Initialize class
    planet = utils.initialize_model(
        opacity,
        atm,
        phase=phase*np.pi/180.0,
        num_gangle=8, 
        num_tangle=8,
        surface_albedo=10.0**log10_As,
        stellar_teff=5780.0,
        stellar_metallicity=0.0,
        stellar_logg=4.0,
        semi_major=a,
        stellar_radius=1.0,
        planet_radius=10.0**log10_Rp,
        planet_mass=10.0**log10_Mp,
        p_reference=np.minimum(1.0, P_surf),
        cloud_frac=None,
        cloud_df=None
    )

    # Compute spectrum
    df = utils.spectrum(planet, opacity, cloud_df, haze_df, water_cloud_frac=10.0**log10_fc)

    # Unpack result
    wv = 1e4/df['wavenumber'][::-1].copy()
    albedo = df['albedo'][::-1].copy()
    fpfs = df['fpfs_reflected'][::-1].copy()

    if R is not None:
        wavl = stars.make_bins(wv)
        wavl_new = stars.grid_at_resolution(wavl[0], wavl[-1], R)
        wv_new = (wavl_new[1:] + wavl_new[:-1])/2
        albedo_new = stars.rebin(wavl, albedo, wavl_new)
        fpfs_new = stars.rebin(wavl, fpfs, wavl_new)

        wv = wv_new
        albedo = albedo_new
        fpfs = fpfs_new

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

def model(x, opacity, wv_bins):

    wv, albedo, fpfs1 = model_raw(x, opacity)
    wavl = stars.make_bins(wv)

    fpfs = np.empty(len(wv_bins))
    for i,b in enumerate(wv_bins):
        fpfs[i] = stars.rebin(wavl.copy(), fpfs1.copy(), b.copy())
    
    return fpfs

