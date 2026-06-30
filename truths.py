import numpy as np
import pickle
import utils
from picaso import justdoit as jdi
from picaso.experimental import interface
from picaso.experimental import utils as eutils
import pandas as pd
from photochem.utils import stars
from model import model_raw, get_result

def get_elizabeth_atm():
    with open('data/PhotochemPT_MiniNep_2.0_2.0_50.0_1.0_0.7525_5.0.pkl','rb') as f:
        res = pickle.load(f)
    sol = res['sol_dict']
    sol['pressure'] /= 1e6
    for key in sol:
        sol[key] = sol[key][::-1].copy()
    sol_new = {}
    for key in sol:
        if 'aer' in key:
            continue
        sol_new[key] = sol[key]
    
    return sol_new

def get_spectrum(df):

    wno, albedo, fpfs = df['wavenumber'], df['albedo'], df['fpfs_reflected']
    wv = 1e4/wno[::-1].copy()
    albedo = albedo[::-1].copy()
    fpfs = fpfs[::-1].copy()
    wavl = stars.make_bins(wv)

    return wv, wavl, albedo, fpfs


def _pressure_weighted_mean(atm, key, pmax=100.0):
    """Pressure-weighted mean from TOA down to pmax (bar)."""
    if key not in atm:
        return 0.0
    p = np.asarray(atm["pressure"], dtype=float)
    x = np.asarray(atm[key], dtype=float)
    mask = np.isfinite(p) & np.isfinite(x) & (p <= pmax)
    if not np.any(mask):
        raise ValueError(f"No finite values found for {key} below {pmax} bar")
    p = p[mask]
    x = x[mask]
    order = np.argsort(p)
    p = p[order]
    x = x[order]
    if p.size == 1:
        return float(x[0])
    return float(np.trapz(x, p) / np.trapz(np.ones_like(p), p))


def build_effective_truth(
    atm,
    surface_albedo=10.0**-2.0,
    planet_radius=2.0,
    planet_mass=4.66,
    cloud_log10_P_bottom=np.log10(0.0404),
    cloud_log10_P_thick=np.log10(0.7) - np.log10(0.6),
    cloud_opd=10.0,
    cloud_frac=0.5,
    semi_major=1.0,
    phase=90.0,
    surface_pressure=100.0,
    surface_pressure_mixing_ratio_est=100.0,
    temperature_override=None,
):
    """Construct a 19-parameter retrieval-space truth vector.

    The vector is a proxy truth in the retrieval parameterization, not a literal
    one-to-one mapping of the physical photochemical atmosphere.
    """
    truth = np.zeros(19, dtype=float)
    pmax = np.minimum(surface_pressure_mixing_ratio_est, surface_pressure)
    cloud_top_pressure = 10.0 ** cloud_log10_P_bottom * 10.0 ** (-cloud_log10_P_thick)
    cloud_thickness = 10.0 ** cloud_log10_P_bottom - cloud_top_pressure

    truth[0] = _pressure_weighted_mean(atm, "temperature", pmax=pmax)
    if temperature_override is not None:
        truth[0] = temperature_override
    truth[1] = np.log10(surface_albedo)
    truth[2] = np.log10(cloud_top_pressure)
    truth[3] = np.log10(cloud_thickness)
    truth[4] = np.log10(cloud_opd)
    truth[5] = cloud_frac
    truth[6] = np.log10(planet_radius)
    truth[7] = np.log10(planet_mass)
    truth[8] = semi_major
    truth[9] = phase
    truth[10] = np.log10(surface_pressure)

    n2 = _pressure_weighted_mean(atm, "N2", pmax=pmax)
    h2 = _pressure_weighted_mean(atm, "H2", pmax=pmax)
    bg_total = n2 + h2
    if bg_total > 0.0:
        truth[11] = h2 / bg_total
    else:
        truth[11] = 0.5

    trace_species = ["O2", "H2O", "CO2", "CH4", "O3", "CO", "NH3"]
    for i, sp in enumerate(trace_species, start=12):
        truth[i] = np.log10(max(_pressure_weighted_mean(atm, sp, pmax=pmax), 1e-300))
        truth[i] = np.maximum(truth[i], -10.0)

    return truth


def _build_neptune_truth():
    atm = pd.DataFrame(get_elizabeth_atm())
    return build_effective_truth(
        atm,
        surface_albedo=10.0**-2.0,
        planet_radius=2.0,
        planet_mass=4.66,
        cloud_log10_P_bottom=np.log10(0.0404),
        cloud_log10_P_thick=np.log10(0.7) - np.log10(0.6),
        cloud_opd=10.0,
        cloud_frac=0.5,
        semi_major=1.0,
        phase=90.0,
        surface_pressure=1000.0,
        surface_pressure_mixing_ratio_est=1.0,
        temperature_override=300,
    )


def _build_archean_truth():
    mix = {
        "N2": 0.945,
        "CO2": 0.05,
        "CO": 0.0005,
        "CH4": 0.005,
        "H2O": 0.003,
    }
    ftot = sum(mix.values())
    for key in mix:
        mix[key] /= ftot

    pressure = np.logspace(-8, np.log10(1), 100)
    atm = jdi.inputs().TP_line_earth(pressure)
    for key in mix:
        atm[key] = np.ones_like(pressure) * mix[key]
    atm = pd.DataFrame(atm)

    return build_effective_truth(
        atm,
        surface_albedo=0.05,
        planet_radius=1.0,
        planet_mass=1.0,
        cloud_log10_P_bottom=np.log10(0.7),
        cloud_log10_P_thick=np.log10(0.7) - np.log10(0.6),
        cloud_opd=10.0,
        cloud_frac=0.5,
        semi_major=1.0,
        phase=90.0,
        surface_pressure=1.0,
        surface_pressure_mixing_ratio_est=1.0,
        temperature_override=None,
    )

def _build_superarchean_truth():
    truth = _build_archean_truth()
    planet_radius = 1.3
    planet_mass = utils.inverse_zeng_Mp_Rp_relation(planet_radius, 0.33)
    truth[6] = np.log10(planet_radius)
    truth[7] = np.log10(planet_mass)
    return truth

def neptune_spectra(opacity):

    atm = get_elizabeth_atm()
    atm = pd.DataFrame(atm)

    planet_radius=2.0
    planet_mass=4.66

    # Water cloud
    # pbot = 0.0348
    pbot = 0.0404
    cloud_log10_P_bottom = np.log10(pbot)
    cloud_log10_P_thick = np.log10(0.7) - np.log10(0.6)
    cloud_df = utils.build_cloud_df(
        atm,
        cloud_scheme="picaso",
        cloud_opd=10.0,
        cloud_w0=0.99,
        cloud_g0=0.85,
        cloud_log10_P_bottom=cloud_log10_P_bottom,
        cloud_log10_P_thick=cloud_log10_P_thick,
    )

    # Initialize
    planet = utils.initialize_model(
        opacity,
        atm,
        phase=90.0*np.pi/180,
        num_gangle=4, 
        num_tangle=4,
        surface_albedo=None,
        semi_major=1.0,
        stellar_radius=1.0,
        planet_radius=planet_radius,
        planet_mass=planet_mass,
        cloud_frac=0.5,
        cloud_df=cloud_df
    )

    df = planet.spectrum(opacity, calculation='reflected')

    return get_result(df, opacity)

def archean_spectra(opacity):
    planet_radius = 1.0 
    planet_mass = 1.0
    return archean_spectra_base(opacity, planet_radius, planet_mass)

def superarchean_spectra(opacity):
    planet_radius = 1.3
    planet_mass = utils.inverse_zeng_Mp_Rp_relation(planet_radius, 0.33)
    return archean_spectra_base(opacity, planet_radius, planet_mass)

def archean_spectra_base(opacity, planet_radius, planet_mass):

    mix = {
        "N2": 0.945,
        "CO2": 0.05,
        "CO": 0.0005,
        "CH4": 0.005,
        "H2O": 0.003
    }
    ftot = sum(mix.values())
    for key in mix:
        mix[key] /= ftot
    P = np.logspace(-8, np.log10(1), 100)
    atm = jdi.inputs().TP_line_earth(P)
    for key in mix:
        atm[key] = np.ones_like(P)*mix[key]
    atm = pd.DataFrame(atm)

    # Water cloud
    ptop = 0.6
    pbot = 0.7
    cloud_log10_P_thick = np.log10(pbot) - np.log10(ptop)  
    cloud_log10_P_bottom = np.log10(pbot)
    cloud_df = utils.build_cloud_df(
        atm,
        cloud_scheme="picaso",
        cloud_opd=10.0,
        cloud_w0=0.99,
        cloud_g0=0.85,
        cloud_log10_P_bottom=cloud_log10_P_bottom,
        cloud_log10_P_thick=cloud_log10_P_thick,
    )

    planet = utils.initialize_model(
        opacity,
        atm,
        phase=90.0*np.pi/180.0,
        num_gangle=4, 
        num_tangle=4,
        surface_albedo=0.05,
        stellar_teff=5778.0,
        stellar_metallicity=0.0,
        stellar_logg=4.4,
        semi_major=1.0,
        stellar_radius=1.0,
        planet_radius=planet_radius,
        planet_mass=planet_mass,
        cloud_frac=None,
        cloud_df=None
    )

    df = planet.spectrum(opacity, calculation='reflected')

    return get_result(df, opacity)

def neptune_spectra_in_model(opacity):
    x = _build_neptune_truth()
    return model_raw(x, opacity)

def archean_spectra_in_model(opacity):
    x = _build_archean_truth()
    return model_raw(x, opacity)

def superarchean_spectra_in_model(opacity):
    x = _build_superarchean_truth()
    return model_raw(x, opacity)

def make_data_from_spectrum(bin_edges, fpfs, bins_data, snr, fpfs_signal, truth):

    fpfs_data = eutils.rebin_edges(bin_edges, fpfs, bins_data)
    fpfs_err_data = np.ones(bins_data.shape[0])*fpfs_signal/snr
    wv_data = (bins_data[:,0] + bins_data[:,1])/2
    wv_err_data = (bins_data[:,1] - bins_data[:,0])/2

    data_dict = {}
    data_dict['bins'] = bins_data
    data_dict['fpfs'] = fpfs_data
    data_dict['err'] = fpfs_err_data
    data_dict['wv'] = wv_data
    data_dict['wv_err'] = wv_err_data
    data_dict['truth'] = truth

    return data_dict

def make_data():

    snr = 10.0
    fpfs_signal = 4.0e-10

    opacity_gap = interface.opannection(
        filename_db='picasofiles/opacities_ck_gap.h5'
    )
    opacity_nogap = interface.opannection(
        filename_db='picasofiles/opacities_ck_nogap.h5'
    )

    bin_edges1 = np.array([[0.45, 0.55]])
    wavelength_edges = eutils.grid_near_resolution(0.83, 1.0, 140.0)
    bin_edges2 = eutils.bin_edges_from_wavelength_edges(wavelength_edges)
    bin_data_gap = np.concatenate((bin_edges1, bin_edges2))

    wavelength_edges = eutils.grid_near_resolution(0.45, 1.0, 140.0)
    bin_data_nogap = eutils.bin_edges_from_wavelength_edges(wavelength_edges)

    data = {}

    # Neptune nogap
    bin_edges, fpfs, albedo = neptune_spectra_in_model(opacity_gap)
    truth = _build_neptune_truth()
    data_dict = make_data_from_spectrum(bin_edges, fpfs, bin_data_gap, snr, fpfs_signal, truth)
    data['neptune_gap'] = data_dict

    # Neptune gap
    bin_edges, fpfs, albedo = neptune_spectra_in_model(opacity_nogap)
    truth = _build_neptune_truth()
    data_dict = make_data_from_spectrum(bin_edges, fpfs, bin_data_nogap, snr, fpfs_signal, truth)
    data['neptune_nogap'] = data_dict

    # Archean nogap
    bin_edges, fpfs, albedo = archean_spectra_in_model(opacity_gap)
    truth = _build_archean_truth()
    data_dict = make_data_from_spectrum(bin_edges, fpfs, bin_data_gap, snr, fpfs_signal, truth)
    data['archean_gap'] = data_dict

    # Archean gap
    bin_edges, fpfs, albedo = archean_spectra_in_model(opacity_nogap)
    truth = _build_archean_truth()
    data_dict = make_data_from_spectrum(bin_edges, fpfs, bin_data_nogap, snr, fpfs_signal, truth)
    data['archean_nogap'] = data_dict

    # Super Archean nogap
    bin_edges, fpfs, albedo = superarchean_spectra_in_model(opacity_gap)
    truth = _build_superarchean_truth()
    data_dict = make_data_from_spectrum(bin_edges, fpfs, bin_data_gap, snr, fpfs_signal, truth)
    data['superarchean_gap'] = data_dict

    # Super Archean gap
    bin_edges, fpfs, albedo = superarchean_spectra_in_model(opacity_nogap)
    truth = _build_superarchean_truth()
    data_dict = make_data_from_spectrum(bin_edges, fpfs, bin_data_nogap, snr, fpfs_signal, truth)
    data['superarchean_nogap'] = data_dict

    return data
