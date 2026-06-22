import numpy as np
from matplotlib import pyplot as plt
import utils
from picaso import justdoit as jdi
from picaso.experimental import interface
import numba as nb
from picaso.experimental import utils as eutils

def main():
    opacity = interface.opannection(
        filename_db='picasofiles/opacities.h5',
    )

    mix = {
        'N2': 0.78,
        'O2': 0.21,
        'H2O': 3.0e-3,
        'CO2': 4.0e-4,
        'CH4': 2.0e-6,
        'O3': 7.0e-7,
    }
    mix['Ar'] = np.maximum(1.0 - sum(mix.values()), 1.0e-100)
    atm = utils.build_atmosphere(mix, T=255.0, log10_P_surf=np.log10(1.0), log10_P_top=-6.0, nlevels=100)
    cloud_df = utils.build_cloud_df(
        atm,
        cloud_scheme="rfast-water",
        cloud_top_pressure=0.6,
        cloud_thickness=0.1,
        cloud_opd=10.0,
        cloud_opdir="data/hires_opacities",
        cloud_lamc0=0.55,
    )

    planet = utils.initialize_model(
        opacity,
        atm,
        phase=90.0*np.pi/180.0,
        num_gangle=8, 
        num_tangle=8,
        surface_albedo=0.05,
        semi_major=1.0,
        stellar_radius=1.0,
        planet_radius=1.0,
        planet_mass=1.0,
        cloud_frac=0.5,
        cloud_df=cloud_df,
    )

    df = planet.spectrum(opacity, calculation='reflected')

    # Unpack result
    result = opacity.rad.reflected_result
    wavl = eutils.grid_near_resolution(0.45, 1.8, 140.0)
    fpfs = eutils.rebin(
        eutils.wavelength_edges_from_bin_edges(result.bin_edges),
        result.fpfs,
        wavl
    )

    fig, ax = plt.subplots(1,1,figsize=[5,4])
    ax.stairs(fpfs, wavl)
    ax.set_ylim(0, 1.75e-10)
    ax.set_xlim(0.4, 1.8)
    ax.set_xlabel('Wavelength (micron)')
    ax.set_ylabel('Planet to star flux ratio')

    plt.savefig('figures/salvador2023_fig1.pdf',bbox_inches='tight')

if __name__ == '__main__':
    nb.set_num_threads(4)
    main()
