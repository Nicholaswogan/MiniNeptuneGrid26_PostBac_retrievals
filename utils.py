import warnings
warnings.filterwarnings('ignore')
import input_files

from copy import deepcopy
import astropy.units as u
import numpy as np
import pandas as pd
from picaso import atmsetup, justdoit as jdi
from photochem.utils import stars

_ATMSETUP_WEIGHT_HELPER = atmsetup.ATMSETUP.__new__(atmsetup.ATMSETUP)

def build_atmosphere(mix, T, log10_P_surf, log10_P_top, nlevels):

    mix_copy = deepcopy(mix)
    # Normalize mix
    ftot = sum(mix_copy.values())
    for key in mix_copy:
        mix_copy[key] /= ftot

    atm = {
        'pressure': np.logspace(log10_P_top, log10_P_surf, nlevels),
        'temperature': np.ones(nlevels)*T,
    }
    for key in mix_copy:
        atm[key] = np.ones(nlevels)*mix_copy[key]

    return pd.DataFrame(atm)


def compute_column_mixing_ratios(df):
    """
    Return the column mixing ratio of each species in a pressure-edge profile.

    The result is the pressure-weighted average of the local volume mixing ratio,
    with the mean molecular weight computed from PICASO's `get_weights`.

    Assumptions
    -----------
    - `df` is ordered from TOA to surface, so pressure increases monotonically.
    - Species columns are local volume mixing ratios at level edges.
    - `species_cols` contains the full composition needed to compute mubar well.

    Parameters
    ----------
    df : pandas.DataFrame
        Atmosphere profile with pressure and species columns.
    pressure_col : str
        Name of the pressure column.
    species_cols : list[str] or None
        Species columns to include. If None, all columns except pressure and
        temperature are used.

    Returns
    -------
    dict
        Column mixing ratio for each species.
    """

    pressure_col = 'pressure'
    if pressure_col not in df.columns:
        raise ValueError(f"Missing pressure column '{pressure_col}'")
    
    species_cols = [
        c for c in df.columns
        if c not in {pressure_col, "temperature"}
    ]

    if len(species_cols) == 0:
        raise ValueError("No species columns were provided or inferred.")

    pressure = np.asarray(df[pressure_col], dtype=float)
    if np.any(np.diff(pressure) <= 0):
        raise ValueError("Pressure must increase monotonically from TOA to surface.")

    weights = _ATMSETUP_WEIGHT_HELPER.get_weights(species_cols)
    mol_weights = np.asarray([weights[s] for s in species_cols], dtype=float)

    x_edge = df[species_cols].to_numpy(dtype=float)
    x_layer = 0.5 * (x_edge[:-1, :] + x_edge[1:, :])

    mu_layer = np.sum(x_layer * mol_weights[None, :], axis=1)
    if np.any(mu_layer <= 0):
        raise ValueError("Computed mean molecular weight is non-positive in at least one layer.")

    dp = np.diff(pressure)
    layer_weight = dp / mu_layer
    denom = np.sum(layer_weight)
    if denom == 0:
        raise ValueError("Pressure grid has zero total span.")

    return {
        species: np.sum(x_layer[:, i] * layer_weight) / denom
        for i, species in enumerate(species_cols)
    }


def _build_rfast_style_cloud_profile(atm, cloud_top_pressure, cloud_thickness, cloud_opd, cloud_w0, cloud_g0):
    """
    Build a PICASO cloud dataframe that mimics RFAST's uniform-in-pressure cloud slab.

    The cloud is specified by:
    - cloud_top_pressure: top of cloud in bar
    - cloud_thickness: total pressure thickness of the cloud slab in bar
    - cloud_opd: total optical depth integrated across the full cloud slab

    Each atmospheric layer receives the fraction of `cloud_opd` corresponding to the
    overlap of that layer with the slab.
    """
    if cloud_thickness <= 0:
        raise ValueError("cloud_thickness must be positive")
    if cloud_top_pressure <= 0:
        raise ValueError("cloud_top_pressure must be positive")

    pressure_edges = np.asarray(atm["pressure"], dtype=float)
    if np.any(np.diff(pressure_edges) <= 0):
        raise ValueError("Atmospheric pressure must increase monotonically from TOA to surface.")

    layer_top = pressure_edges[:-1]
    layer_bottom = pressure_edges[1:]
    layer_pressure = np.sqrt(layer_top * layer_bottom)

    cloud_bottom_pressure = cloud_top_pressure + cloud_thickness
    overlap = np.maximum(
        0.0,
        np.minimum(layer_bottom, cloud_bottom_pressure) - np.maximum(layer_top, cloud_top_pressure),
    )
    layer_opd = cloud_opd * (overlap / cloud_thickness)

    wno = np.asarray(jdi.get_cld_input_grid("wave_EGP.dat"), dtype=float)
    pressure_all = np.repeat(layer_pressure, len(wno))
    wno_all = np.tile(wno, len(layer_pressure))
    opd_all = np.repeat(layer_opd, len(wno))
    w0_all = np.full_like(opd_all, cloud_w0, dtype=float)
    g0_all = np.full_like(opd_all, cloud_g0, dtype=float)

    return pd.DataFrame(
        {
            "pressure": pressure_all,
            "wavenumber": wno_all,
            "opd": opd_all,
            "w0": w0_all,
            "g0": g0_all,
        }
    )


def initialize_model(
    opacity,
    atm,
    phase=0.0,
    num_gangle=8, 
    num_tangle=8,
    surface_albedo=0.05,
    stellar_teff=5780.0,
    stellar_metallicity=0.0,
    stellar_logg=4.0,
    semi_major=1.0,
    stellar_radius=1.0,
    planet_radius=1.0,
    planet_mass=1.0,
    cloud_frac=0.5,
    cloud_scheme="rfast",
    cloud_top_pressure=0.6,
    cloud_thickness=0.1,
    cloud_opd=10.0,
    cloud_w0=0.99,
    cloud_g0=0.85,
    cloud_log10_P_bottom=None,
    cloud_log10_P_thick=None,
    clouds=True
):
    
    # Initialize
    earth = jdi.inputs()

    # Set phase
    earth.phase_angle(phase, num_gangle=num_gangle, num_tangle=num_tangle)
    
    # Gravity
    earth.gravity(
        radius=planet_radius,
        radius_unit=u.Unit("R_earth"),
        mass=planet_mass,
        mass_unit=u.Unit("M_earth"),
    )

    # Star
    earth.star(
        opannection=opacity,
        temp=stellar_teff,
        metal=stellar_metallicity,
        logg=stellar_logg,
        semi_major=semi_major,
        radius=stellar_radius,
        radius_unit=u.Unit("R_sun"),
        semi_major_unit=u.Unit("au"),
    )

    # Set surface albedo
    if surface_albedo is not None:
        earth.surface_reflect(surface_albedo, opacity.wno)

    # Set atmosphere
    earth.atmosphere(df=atm)

    # Set the cloud
    weight_clear = 1.0 - cloud_frac
    if clouds:
        if cloud_scheme == "rfast":
            if cloud_top_pressure is None or cloud_thickness is None:
                raise ValueError(
                    "cloud_top_pressure and cloud_thickness must both be provided "
                    "for cloud_scheme='rfast'"
                )
            cloud_df = _build_rfast_style_cloud_profile(
                atm=atm,
                cloud_top_pressure=cloud_top_pressure,
                cloud_thickness=cloud_thickness,
                cloud_opd=cloud_opd,
                cloud_w0=cloud_w0,
                cloud_g0=cloud_g0,
            )
            earth.clouds(
                df=cloud_df,
                do_holes=True,
                fhole=weight_clear,
                fthin_cld=0.0,
            )
        elif cloud_scheme == "picaso":
            if cloud_log10_P_bottom is None or cloud_log10_P_thick is None:
                raise ValueError(
                    "cloud_log10_P_bottom and cloud_log10_P_thick must both be provided "
                    "for cloud_scheme='picaso'"
                )
            earth.clouds(
                g0=[cloud_g0],
                w0=[cloud_w0],
                opd=[cloud_opd],
                p=[cloud_log10_P_bottom],
                dp=[cloud_log10_P_thick],
                do_holes=True,
                fhole=weight_clear,
                fthin_cld=0.0,
            )
        else:
            raise ValueError("cloud_scheme must be either 'rfast' or 'picaso'")

    return earth


def initialize_model_rfast(
    opacity,
    atm,
    **kwargs,
):
    """
    Convenience wrapper for the RFAST-style cloud parameterization.
    """
    kwargs["cloud_scheme"] = "rfast"
    return initialize_model(opacity=opacity, atm=atm, **kwargs)


def initialize_model_picaso(
    opacity,
    atm,
    **kwargs,
):
    """
    Convenience wrapper for the legacy PICASO-style cloud parameterization.
    """
    kwargs["cloud_scheme"] = "picaso"
    return initialize_model(opacity=opacity, atm=atm, **kwargs)

def quantile_to_uniform(quantile, lower_bound, upper_bound):
    return quantile*(upper_bound - lower_bound) + lower_bound


def model_raw(x, opacity, R=None):
    T = x[0]
    log10_As = x[1]
    log10_pc = x[2]
    log10_dpc = x[3]
    log10_tauc = x[4]
    log10_fc = x[5]
    log10_Rp = x[6]
    log10_Mp = x[7]
    a = x[8] # in AU
    phase = x[9] # degrees
    log10Pi = x[10:]

    species = ['N2', 'O2', 'H2O', 'CO2', 'CH4', 'O3', 'H2', 'CO']
    Pi = 10.0**(log10Pi)
    P_surf = np.sum(Pi)
    mix_i = P_surf/Pi
    mix = {}
    for i,sp in enumerate(species):
        mix[sp] = mix_i[i]

    atm = build_atmosphere(mix, T, np.log10(P_surf), log10_P_top=-6.0, nlevels=70)

    planet = initialize_model(
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
        cloud_frac=10.0**log10_fc,
        cloud_top_pressure=10.0**log10_pc,
        cloud_thickness=10.0**log10_dpc,
        cloud_opd=10.0**log10_tauc,
        cloud_w0=0.99,
        cloud_g0=0.85,
        clouds=True
    )

    df = planet.spectrum(opacity, calculation='reflected')

    wv = 1e4/df['wavenumber'][::-1].copy()
    albedo = df['albedo'][::-1].copy()
    fpfs = df['fpfs_reflected'][::-1].copy()

    if R is not None:
        wavl = stars.make_bins(wv)
        wavl_new = stars.grid_at_resolution(wavl[0], wavl[1], R)
        wv_new = (wavl_new[1:] + wavl[:-1])/2
        albedo_new = stars.rebin(wavl, albedo, wavl_new)
        fpfs_new = stars.rebin(wavl, fpfs, wavl_new)

        wv = wv_new
        albedo = albedo_new
        fpfs = fpfs_new

    return wv, albedo, fpfs

def model(x, opacity, wv_bins):

    wv, albedo, fpfs1 = model_raw(x, opacity)
    wavl = stars.make_bins(wv)

    fpfs = np.empty(len(wv_bins))
    for i,b in enumerate(wv_bins):
        fpfs[i] = stars.rebin(wavl, fpfs1, b)
    
    return fpfs

