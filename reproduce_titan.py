
import numpy as np
import re
from matplotlib import pyplot as plt
import utils
from picaso import justdoit as jdi
import numba as nb
import haze


_MCKAY_PT_PROFILE_CODE = """
      DATA TLE/   ! TEMPERATURE IN k
     & 183.0, 181.0, 178.0, 174.7, 169.0, 164.0, 158.0, 150.0, 147.0,
     & 136.0, 135.0, 137.0, 143.0, 146.5, 150.0, 154.5, 159.0, 163.5,
     & 166.5, 169.0, 171.1, 172.8, 174.3, 175.1, 175.6, 175.9, 176.0,
     & 176.0, 175.9, 175.8, 175.7, 175.6, 175.5, 175.4, 175.3, 175.2, 
     & 175.1, 175.0, 174.9, 174.8, 174.6, 174.4,
     & 174.0, 173.9, 173.8, 173.6, 173.4, 173.2, 173.0, 172.9, 172.7,
     & 172.5, 172.4, 172.2, 172.0, 171.8, 171.6, 171.4, 171.2, 171.0,
     & 170.8, 170.4, 169.9, 169.5, 169.0, 168.5, 167.9, 167.4, 166.7,
     & 166.0, 165.3, 164.5, 163.7, 163.0, 162.4, 161.8, 161.2, 160.5,
     & 159.6, 158.6, 157.5, 156.3, 155.3, 154.3, 153.4, 152.6, 151.9,
     & 151.0, 150.1, 149.0, 147.8, 146.5, 145.1, 143.7, 142.1, 140.6,
     & 139.1, 137.6, 135.8, 133.8, 131.5, 128.8, 125.8, 122.7, 119.7,
     & 116.6, 111.9, 106.1, 100.5,  92.6,  85.9,  80.9,  77.6,  75.4,
     &  73.8,  72.8,  72.2,  71.7,  71.5,  71.4,  71.2,  71.1,  71.1,
     &  71.2,  71.4,  71.5,  71.8,  72.3,  73.1,  73.9,  74.7,  75.7,
     &  76.7,  78.0,  79.2,  80.6,  82.1,  83.6,  85.5,  87.3,  88.5,
     &  89.5,  90.5,  91.5,  92.1,  92.7,  93.3,  93.9/
      DATA PLE/  ! PRESSURE IN MILLIBARS
     & 8.08e-09, 1.39e-08, 2.45e-08, 4.41e-08, 1.03e-07, 2.04e-07,
     & 4.27e-07, 9.44e-07, 1.32e-06, 3.92e-06, 5.77e-06, 1.26e-05,
     & 4.11e-05, 6.68e-05, 1.08e-04, 1.75e-04, 2.80e-04, 4.46e-04,
     & 7.08e-04, 1.12e-03, 1.78e-03, 2.84e-03, 4.54e-03, 6.03e-03,
     & 8.01e-03, 9.70e-03, 1.18e-02, 1.43e-02, 1.73e-02, 3.13e-02,
     & 3.83e-02, 4.69e-02, 5.74e-02, 7.05e-02, 8.66e-02, 1.32e-01,
     & 2.01e-01, 2.49e-01, 3.09e-01, 3.84e-01, 4.78e-01, 5.96e-01, !
     & 7.59E-01, 7.98E-01, 8.37E-01, 8.77E-01, 9.20E-01, 9.64E-01,
     & 1.01E+00, 1.06E+00, 1.11E+00, 1.16E+00, 1.22E+00, 1.28E+00,
     & 1.35E+00, 1.41E+00, 1.48E+00, 1.55E+00, 1.63E+00, 1.71E+00,
     & 1.79E+00, 1.88E+00, 1.97E+00, 2.07E+00, 2.17E+00, 2.29E+00,
     & 2.40E+00, 2.52E+00, 2.65E+00, 2.78E+00, 2.93E+00, 3.08E+00,
     & 3.24E+00, 3.41E+00, 3.59E+00, 3.78E+00, 3.90E+00, 4.19E+00,
     & 4.42E+00, 4.66E+00, 4.91E+00, 5.19E+00, 5.48E+00, 5.78E+00,
     & 6.10E+00, 6.45E+00, 6.82E+00, 7.22E+00, 7.63E+00, 8.08E+00,
     & 8.56E+00, 9.06E+00, 9.61E+00, 1.02E+01, 1.08E+01, 1.15E+01,
     & 1.22E+01, 1.30E+01, 1.38E+01, 1.47E+01, 1.57E+01, 1.68E+01,
     & 1.80E+01, 1.93E+01, 2.07E+01, 2.23E+01, 2.40E+01, 2.60E+01,
     & 2.83E+01, 3.10E+01, 3.42E+01, 3.79E+01, 4.23E+01, 4.75E+01,
     & 5.34E+01, 6.01E+01, 6.79E+01, 7.67E+01, 8.67E+01, 9.81E+01,
     & 1.11E+02, 1.26E+02, 1.42E+02, 1.61E+02, 1.83E+02, 2.07E+02,
     & 2.35E+02, 2.65E+02, 3.00E+02, 3.40E+02, 3.83E+02, 4.32E+02,
     & 4.87E+02, 5.47E+02, 6.14E+02, 6.88E+02, 7.70E+02, 8.59E+02,
     & 9.57E+02, 1.06E+03, 1.12E+03, 1.18E+03, 1.24E+03, 1.30E+03,
     & 1.34E+03, 1.37E+03, 1.40E+03, 1.44E+03/"""


def _extract_fortran_data_block(text, array_name):
    pattern = rf"DATA\s+{array_name}/(.*?)/"
    match = re.search(pattern, text, flags=re.S | re.I)
    if match is None:
        raise ValueError(f"Could not find DATA block for {array_name} in the embedded McKay profile text")
    block = match.group(1)
    values = re.findall(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?", block)
    return np.asarray([float(v.replace("D", "E").replace("d", "E")) for v in values], dtype=float)


def _load_mckay_lellouch_profile():
    pressure_mbar = _extract_fortran_data_block(_MCKAY_PT_PROFILE_CODE, "PLE")
    temperature_k = _extract_fortran_data_block(_MCKAY_PT_PROFILE_CODE, "TLE")
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
