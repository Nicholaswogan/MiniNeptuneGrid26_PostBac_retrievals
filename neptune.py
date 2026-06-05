import numpy as np
import pickle
import utils
from picaso import justdoit as jdi
import pandas as pd
import haze
from photochem.utils import stars

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


def build_effective_truth(atm, haze_log10_prod, surface_log10_albedo=-2.0):
    """Construct a 20-parameter retrieval-space truth vector.

    The vector is a proxy truth in the retrieval parameterization, not a literal
    one-to-one mapping of the physical photochemical atmosphere.
    """
    p_eff = 100.0
    truth = np.zeros(20, dtype=float)

    truth[0] = _pressure_weighted_mean(atm, "temperature", pmax=p_eff)
    truth[1] = surface_log10_albedo
    truth[2] = np.log10(p_eff)
    truth[3] = np.log10(0.7) - np.log10(0.6)
    truth[4] = np.log10(10.0)
    truth[5] = haze_log10_prod
    truth[6] = np.log10(0.5)
    truth[7] = np.log10(2.0)
    truth[8] = np.log10(4.66)
    truth[9] = 1.0
    truth[10] = 90.0
    truth[11] = np.log10(p_eff)

    n2 = _pressure_weighted_mean(atm, "N2", pmax=p_eff)
    h2 = _pressure_weighted_mean(atm, "H2", pmax=p_eff)
    bg_total = n2 + h2
    if bg_total > 0.0:
        truth[12] = h2 / bg_total
    else:
        truth[12] = 0.5

    trace_species = ["O2", "H2O", "CO2", "CH4", "O3", "CO", "NH3"]
    for i, sp in enumerate(trace_species, start=13):
        truth[i] = np.log10(max(_pressure_weighted_mean(atm, sp, pmax=p_eff), 1e-300))

    return truth

def neptune_spectra():

    opacity = jdi.opannection(
        wave_range=[0.2,1.8],
        filename_db='picasofiles/opacities_photochem_0.1_250.0_R15000_v2.db',
    )

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

    # Haze
    m = haze.McKayTitanHazeModel(sweep_clear_below_pressure=1.0e7)
    solution = m.solve_from_atmosphere(
        atm,
        column_production=3.0e-14,
        peak_pressure=1.0e-6,
        width_pressure=1.0e-6 * np.exp(-1.124),
        planet_radius=planet_radius,
        planet_mass=planet_mass,
        reference_pressure=1.0,
    )
    haze_df = haze.make_picaso_haze_clouddf_from_solution(
        solution, 
        refractive_index_file='data/khare_tholins.refrind'
    )

    # Haze df that is very optically thin
    solution = m.solve_from_atmosphere(
        atm,
        column_production=3.0e-14*1e-100,
        peak_pressure=1.0e-6,
        width_pressure=1.0e-6 * np.exp(-1.124),
        planet_radius=planet_radius,
        planet_mass=planet_mass,
        reference_pressure=1.0,
    )
    haze_clear_df = haze.make_picaso_haze_clouddf_from_solution(
        solution, 
        refractive_index_file='data/khare_tholins.refrind'
    )

    # Initialize
    planet = utils.initialize_model(
        opacity,
        atm,
        phase=90.0*np.pi/180,
        num_gangle=4, 
        num_tangle=4,
        surface_albedo=None,
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

    # Hazy
    df1 = utils.spectrum(planet, opacity, cloud_df, haze_df, water_cloud_frac=0.5)

    # Not hazy
    df2 = utils.spectrum(planet, opacity, cloud_df, haze_clear_df, water_cloud_frac=0.5)

    out = {
        'hazy': get_spectrum(df1),
        'not_hazy': get_spectrum(df2),
    }

    return out

def grid_near_resolution(wv_min, wv_max, R):
    """
    Build wavelength bin edges spanning ``[wv_min, wv_max]`` at a resolving
    power as close as practical to ``R``.

    The grid is log-spaced so the bins have approximately constant resolving
    power across the bandpass, but the final bin is forced to land exactly on
    ``wv_max`` rather than leaving a tiny leftover bin.

    Parameters
    ----------
    wv_min : float
        Lower wavelength edge.
    wv_max : float
        Upper wavelength edge.
    R : float
        Target resolving power.

    Returns
    -------
    numpy.ndarray
        Wavelength bin edges.
    """
    wv_min = float(wv_min)
    wv_max = float(wv_max)
    R = float(R)

    if not np.isfinite(wv_min) or not np.isfinite(wv_max) or not np.isfinite(R):
        raise ValueError("wv_min, wv_max, and R must be finite")
    if wv_min <= 0.0 or wv_max <= 0.0:
        raise ValueError("wv_min and wv_max must be positive")
    if wv_max <= wv_min:
        raise ValueError("wv_max must be larger than wv_min")
    if R <= 0.0:
        raise ValueError("R must be positive")

    # For a log-spaced grid, the number of bins needed to achieve a resolution
    # near R is approximately R * ln(wv_max / wv_min). Rounding keeps the grid
    # close to the requested resolving power while ensuring the last bin is not
    # artificially tiny.
    nbin = max(1, int(np.round(R * np.log(wv_max / wv_min))))
    return np.geomspace(wv_min, wv_max, nbin + 1)


def make_data_from_spectrum(wavl, fpfs, R, wavelength_edges, snr, fpfs_signal):

    assert len(R) == len(snr) == len(wavelength_edges) - 1

    fpfs_err = np.zeros(0, np.float64)
    wavl_data = np.zeros(0, np.float64)
    for i in range(len(R)):
        wavl_tmp = grid_near_resolution(wavelength_edges[i], wavelength_edges[i+1], R[i])
        nbins = len(wavl_tmp) - 1
        if i != len(R) - 1:
            wavl_tmp = wavl_tmp[:-1]
        wavl_data = np.append(wavl_data, wavl_tmp)
        fpfs_err = np.append(fpfs_err, np.ones(nbins)*fpfs_signal/snr[i])

    fpfs_data = stars.rebin(wavl, fpfs, wavl_data)
    wv_data = (wavl_data[1:] + wavl_data[:-1])/2
    wv_err_data = (wavl_data[1:] - wavl_data[:-1])/2
    bins_data = np.empty((len(wv_data),2))
    for i in range(wv_data.shape[0]):
        bins_data[i,:] = np.array([wavl_data[i], wavl_data[i+1]])
    
    data_dict = {}
    data_dict['bins'] = bins_data
    data_dict['fpfs'] = fpfs_data
    data_dict['err'] = fpfs_err
    data_dict['wv'] = wv_data
    data_dict['wv_err'] = wv_err_data
    data_dict['wavl'] = wavl_data

    return data_dict

def make_neptune_data(snr):

    R = np.array([140, 70])
    wavelength_edges = np.array([0.45, 1.0, 1.8])
    snr1 = np.array([snr, snr])
    fpfs_signal = 4.0e-10

    out = neptune_spectra()
    atm_truth = get_elizabeth_atm()

    truth_hazy = build_effective_truth(
        atm_truth,
        haze_log10_prod=np.log10(3.0e-14),
    )
    truth_clear = build_effective_truth(
        atm_truth,
        haze_log10_prod=-22.0,
    )

    _, wavl, _, fpfs = out['not_hazy']
    data_dict_clear = make_data_from_spectrum(wavl, fpfs, R, wavelength_edges, snr1, fpfs_signal)
    data_dict_clear['truth'] = truth_clear

    _, wavl, _, fpfs = out['hazy']
    data_dict_hazy = make_data_from_spectrum(wavl, fpfs, R, wavelength_edges, snr1, fpfs_signal)
    data_dict_hazy['truth'] = truth_hazy

    data = {
        'clear': data_dict_clear,
        'hazy': data_dict_hazy
    }

    with open(f'data/neptune_{snr}.pkl','wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    make_neptune_data(snr=20)