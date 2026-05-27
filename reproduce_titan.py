
import numpy as np
import re
from pathlib import Path
from matplotlib import pyplot as plt
import utils
from picaso import justdoit as jdi
import numba as nb
import haze


_MCKAY_PROFILE_PATH = Path(__file__).resolve().parent / "codex_reference" / "McKay" / "tgmsubs.for"


def _extract_fortran_data_block(text, array_name):
    pattern = rf"DATA\s+{array_name}/(.*?)/"
    match = re.search(pattern, text, flags=re.S | re.I)
    if match is None:
        raise ValueError(f"Could not find DATA block for {array_name} in { _MCKAY_PROFILE_PATH }")
    block = match.group(1)
    values = re.findall(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?", block)
    return np.asarray([float(v.replace("D", "E").replace("d", "E")) for v in values], dtype=float)


def _load_mckay_lellouch_profile():
    text = _MCKAY_PROFILE_PATH.read_text()
    pressure_mbar = _extract_fortran_data_block(text, "PLE")
    temperature_k = _extract_fortran_data_block(text, "TLE")
    if pressure_mbar.shape != temperature_k.shape:
        raise ValueError("McKay pressure and temperature profiles must have the same length")
    pressure_bar = pressure_mbar * 1.0e-3
    order = np.argsort(pressure_bar)
    return pressure_bar[order], temperature_k[order]


MCKAY_LELLOUCH_PRESSURE_BAR, MCKAY_LELLOUCH_TEMPERATURE_K = _load_mckay_lellouch_profile()

TITAN_RADIUS_CM = 2575.0e5
TITAN_MASS_G = 1.3452e26
TITAN_RADIUS_REARTH = 2575.0 / 6371.0
TITAN_MASS_MEARTH = 1.3452e26 / 5.9722e27
TITAN_REFERENCE_PRESSURE_BAR = float(MCKAY_LELLOUCH_PRESSURE_BAR[-1])
TITAN_SEMI_MAJOR_AU = 9.537

MCKAY_COLUMN_PRODUCTION = 0.35 * 3.5e-14
MCKAY_PEAK_PRESSURE_BAR = 1.0e-7
MCKAY_WIDTH_PRESSURE_BAR = 1.0e-7 * np.exp(-1.124)


def mckay_titan_temperature_profile(pressure_bar):
    """Interpolate McKay's Lellouch Titan temperature profile onto a pressure grid."""
    pressure_bar = np.asarray(pressure_bar, dtype=float)
    if np.any(pressure_bar <= 0.0):
        raise ValueError("`pressure_bar` must be positive")
    log_p = np.log10(pressure_bar)
    log_p_ref = np.log10(MCKAY_LELLOUCH_PRESSURE_BAR)
    return np.interp(
        log_p,
        log_p_ref,
        MCKAY_LELLOUCH_TEMPERATURE_K,
        left=MCKAY_LELLOUCH_TEMPERATURE_K[0],
        right=MCKAY_LELLOUCH_TEMPERATURE_K[-1],
    )


def main():
    opacity = jdi.opannection(
        wave_range=[0.2,1.0],
        filename_db='picasofiles/opacities_photochem_0.1_250.0_R15000_v2.db',
    )

    mix = {
        'N2': 0.95,
        'CH4': 0.05,
    }
    ftot = sum(mix.values())
    for key in mix:
        mix[key] /= ftot
    pressure_top = MCKAY_LELLOUCH_PRESSURE_BAR[0]
    pressure_bottom = MCKAY_LELLOUCH_PRESSURE_BAR[-1]
    atm = utils.build_atmosphere(
        mix,
        T=1.0,
        log10_P_surf=np.log10(pressure_bottom),
        log10_P_top=np.log10(pressure_top),
        nlevels=100,
    )
    atm['temperature'] = mckay_titan_temperature_profile(atm['pressure'].to_numpy())

    m = haze.McKayTitanHazeModel()
    solution = m.solve_from_atmosphere(
        atm,
        column_production=MCKAY_COLUMN_PRODUCTION,
        peak_pressure=MCKAY_PEAK_PRESSURE_BAR,
        width_pressure=MCKAY_WIDTH_PRESSURE_BAR,
        planet_radius=TITAN_RADIUS_REARTH,
        planet_mass=TITAN_MASS_MEARTH,
        reference_pressure=TITAN_REFERENCE_PRESSURE_BAR,
    )
    haze_df = haze.make_picaso_haze_clouddf_from_solution(
        solution,
        refractive_index_file='data/khare_tholins.refrind',
    )

    planet = utils.initialize_model(
        opacity,
        atm,
        phase=0.0,
        num_gangle=8, 
        num_tangle=1,
        surface_albedo=0.1,
        stellar_teff=5780.0,
        stellar_metallicity=0.0,
        stellar_logg=4.0,
        semi_major=TITAN_SEMI_MAJOR_AU,
        stellar_radius=1.0,
        planet_radius=TITAN_RADIUS_REARTH,
        planet_mass=TITAN_MASS_MEARTH,
        cloud_frac=None,
        cloud_df=None,
    )

    planet.clouds(df=haze_df)

    df = planet.spectrum(opacity, calculation='reflected')
    wno, albedo = jdi.mean_regrid(df['wavenumber'], df['albedo'], R=140)

    fig, ax = plt.subplots(1,1,figsize=[5,4])
    ax.plot(1e4/wno, albedo)

    wv, alb = np.loadtxt('data/titan_geometric_albedo_neff1984.txt', skiprows=2).T
    ax.plot(wv, alb*(2560/2850)**2, c='k', label='Neff+1984 (measured)')

    ax.set_ylim(0, 0.4)
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel('Wavelength (micron)')
    ax.set_ylabel('Albedo')

    plt.savefig('figures/mckay1989_fig5.pdf',bbox_inches='tight')

if __name__ == '__main__':
    nb.set_num_threads(4)
    main()
