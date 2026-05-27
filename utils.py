import warnings
warnings.filterwarnings('ignore')
import input_files

import os
from astropy.io import ascii
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


def _interp_extrap(x, xp, fp):
    """Linearly interpolate and extrapolate onto x."""
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    y = np.interp(x, xp, fp)

    if xp.size > 1:
        left = x < xp[0]
        right = x > xp[-1]

        if np.any(left):
            slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
            y[left] = fp[0] + slope * (x[left] - xp[0])

        if np.any(right):
            slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
            y[right] = fp[-1] + slope * (x[right] - xp[-1])

    return y


def _read_rfast_cloud_optics(wavelength_um, opdir, lamc0=0.55):
    """
    Read the rfast liquid/ice cloud optical property tables and blend them 50/50.

    Returns wavelength-dependent cloud single-scattering albedo, asymmetry
    parameter, and extinction efficiency normalized at lamc0, following the
    logic in rfast_opac_routines.py.
    """
    liquid_path = os.path.join(opdir, "strato_cum.mie")
    ice_path = os.path.join(opdir, "baum_cirrus_de100.mie")

    if not (os.path.exists(liquid_path) and os.path.exists(ice_path)):
        raise FileNotFoundError(
            "Could not find rfast cloud Mie tables. Expected "
            f"{liquid_path} and {ice_path}."
        )

    liquid = ascii.read(liquid_path, data_start=20, delimiter=" ")
    ice = ascii.read(ice_path, data_start=2, delimiter=" ")

    wcl = _interp_extrap(wavelength_um, liquid["col1"], liquid["col10"])
    gcl = _interp_extrap(wavelength_um, liquid["col1"], liquid["col11"])
    qcl_raw = _interp_extrap(wavelength_um, liquid["col1"], liquid["col7"])
    qcl_ref = _interp_extrap(np.array([lamc0]), liquid["col1"], liquid["col7"])[0]
    qcl = qcl_raw / qcl_ref

    wci = _interp_extrap(wavelength_um, ice["wl"], ice["omega"])
    gci = _interp_extrap(wavelength_um, ice["wl"], ice["g"])
    qci_raw = _interp_extrap(wavelength_um, ice["wl"], ice["Qe"])
    qci_ref = _interp_extrap(np.array([lamc0]), ice["wl"], ice["Qe"])[0]
    qci = qci_raw / qci_ref

    frac_liquid = 0.5
    w0 = frac_liquid * wcl + (1.0 - frac_liquid) * wci
    g0 = frac_liquid * gcl + (1.0 - frac_liquid) * gci
    qext = frac_liquid * qcl + (1.0 - frac_liquid) * qci
    return w0, g0, qext


def _build_rfast_like_cloud_df(pressure_levels_bar, wavenumber, opdir, ptop=0.6,
                               dpc=0.1, tauc0=10.0, lamc0=0.55):
    """
    Build a PICASO cloud dataframe using the same cloud-top pressure and cloud
    thickness convention as RFAST.
    """
    if dpc <= 0:
        raise ValueError("dpc must be positive")

    wavelength_um = 1e4 / np.asarray(wavenumber, dtype=float)

    w0_wave, g0_wave, qext_wave = _read_rfast_cloud_optics(wavelength_um, opdir, lamc0=lamc0)
    
    tau_wave = tauc0 * qext_wave

    level_pressure = np.asarray(pressure_levels_bar, dtype=float)
    layer_pressure = np.sqrt(level_pressure[:-1] * level_pressure[1:])
    pbot = ptop + dpc
    cloud_thickness = dpc

    rows = []
    for ilay, pmid in enumerate(layer_pressure):
        player_top = level_pressure[ilay]
        player_bot = level_pressure[ilay + 1]
        overlap = max(0.0, min(player_bot, pbot) - max(player_top, ptop))
        tau_fraction = overlap / cloud_thickness if cloud_thickness > 0 else 0.0
        opd_layer = tau_wave * tau_fraction

        rows.append(
            pd.DataFrame(
                {
                    "pressure": np.full_like(wavenumber, pmid, dtype=float),
                    "wavenumber": np.asarray(wavenumber, dtype=float),
                    "opd": opd_layer,
                    "w0": w0_wave,
                    "g0": g0_wave,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def build_cloud_df(
    opacity,
    atm,
    cloud_scheme="rfast",
    cloud_top_pressure=0.6,
    cloud_thickness=0.1,
    cloud_opd=10.0,
    cloud_w0=0.99,
    cloud_g0=0.85,
    cloud_opdir=None,
    cloud_lamc0=0.55,
    cloud_log10_P_bottom=None,
    cloud_log10_P_thick=None,
):
    if cloud_scheme == "rfast":
        return _build_rfast_style_cloud_profile(
            atm=atm,
            cloud_top_pressure=cloud_top_pressure,
            cloud_thickness=cloud_thickness,
            cloud_opd=cloud_opd,
            cloud_w0=cloud_w0,
            cloud_g0=cloud_g0,
        )

    if cloud_scheme == "rfast-water":
        if cloud_opdir is None:
            raise ValueError("cloud_opdir must be provided for cloud_scheme='rfast-water'")
        return _build_rfast_like_cloud_df(
            pressure_levels_bar=np.asarray(atm["pressure"], dtype=float),
            wavenumber=opacity.wno,
            opdir=cloud_opdir,
            ptop=cloud_top_pressure,
            dpc=cloud_thickness,
            tauc0=cloud_opd,
            lamc0=cloud_lamc0,
        )

    if cloud_scheme == "picaso":
        if cloud_log10_P_bottom is None or cloud_log10_P_thick is None:
            raise ValueError(
                "cloud_log10_P_bottom and cloud_log10_P_thick must both be provided "
                "for cloud_scheme='picaso'"
            )

        pressure_level = np.asarray(atm["pressure"], dtype=float)
        if np.any(np.diff(pressure_level) <= 0):
            raise ValueError("Atmospheric pressure must increase monotonically from TOA to surface.")

        layer_pressure = np.sqrt(pressure_level[1:] * pressure_level[:-1])
        wno = np.asarray(jdi.get_cld_input_grid("wave_EGP.dat"), dtype=float)

        pressure_all = np.repeat(layer_pressure, len(wno))
        wno_all = np.tile(wno, len(layer_pressure))
        opd_all = np.zeros_like(pressure_all, dtype=float)
        w0_all = np.zeros_like(pressure_all, dtype=float)
        g0_all = np.zeros_like(pressure_all, dtype=float)

        maxp = 10.0 ** cloud_log10_P_bottom
        minp = 10.0 ** (cloud_log10_P_bottom - cloud_log10_P_thick)
        cloud_mask = (pressure_all >= minp) & (pressure_all <= maxp)
        opd_all[cloud_mask] = cloud_opd
        w0_all[cloud_mask] = cloud_w0
        g0_all[cloud_mask] = cloud_g0

        return pd.DataFrame(
            {
                "pressure": pressure_all,
                "wavenumber": wno_all,
                "opd": opd_all,
                "w0": w0_all,
                "g0": g0_all,
            }
        )

    raise ValueError("cloud_scheme must be one of: 'rfast', 'rfast-water', 'picaso'")



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
    cloud_df=None,
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
    if cloud_df is not None:
        weight_clear = 1.0 - cloud_frac
        earth.clouds(
            df=cloud_df,
            do_holes=True,
            fhole=weight_clear,
            fthin_cld=0.0,
        )

    return earth

def spectrum(earth, opacity, water_cloud_df, haze_df, water_cloud_frac=0.5):
    """
    Compute a reflected spectrum with globally uniform haze and patchy water clouds.

    The haze is present in both the clear and cloudy columns. The water cloud is only
    present in the cloudy column, and the final spectrum is the area-weighted mix:

        (1 - water_cloud_frac) * clear_column + water_cloud_frac * cloudy_column

    Parameters
    ----------
    earth : picaso.justdoit.inputs
        Configured PICASO atmosphere/planet object.
    opacity : object
        PICASO opacity connection.
    water_cloud_df : pandas.DataFrame
        Cloud dataframe for the water-cloud column.
    haze_df : pandas.DataFrame
        Cloud dataframe for the globally uniform haze.
    water_cloud_frac : float
        Areal coverage of the cloudy column.
    """
    if not 0.0 <= water_cloud_frac <= 1.0:
        raise ValueError("water_cloud_frac must be between 0 and 1")
    if haze_df is None:
        raise ValueError("haze_df must be provided")
    if water_cloud_df is None:
        raise ValueError("water_cloud_df must be provided")

    def _combine_cloud_dfs(base_df, add_df):
        base = base_df.sort_values(["pressure", "wavenumber"]).reset_index(drop=True)
        add = add_df.sort_values(["pressure", "wavenumber"]).reset_index(drop=True)

        required = {"pressure", "wavenumber", "opd", "w0", "g0"}
        missing_base = required.difference(base.columns)
        missing_add = required.difference(add.columns)
        if missing_base:
            raise ValueError(f"haze_df is missing required columns: {sorted(missing_base)}")
        if missing_add:
            raise ValueError(f"water_cloud_df is missing required columns: {sorted(missing_add)}")

        if base.shape[0] != add.shape[0]:
            raise ValueError("haze_df and water_cloud_df must have the same number of rows")

        if not np.allclose(base["pressure"].to_numpy(dtype=float), add["pressure"].to_numpy(dtype=float)):
            raise ValueError("haze_df and water_cloud_df must use the same pressure grid")
        if not np.allclose(base["wavenumber"].to_numpy(dtype=float), add["wavenumber"].to_numpy(dtype=float)):
            raise ValueError("haze_df and water_cloud_df must use the same wavenumber grid")

        tau_base = base["opd"].to_numpy(dtype=float)
        tau_add = add["opd"].to_numpy(dtype=float)
        w0_base = base["w0"].to_numpy(dtype=float)
        w0_add = add["w0"].to_numpy(dtype=float)
        g0_base = base["g0"].to_numpy(dtype=float)
        g0_add = add["g0"].to_numpy(dtype=float)

        tau_total = tau_base + tau_add
        scat_base = tau_base * w0_base
        scat_add = tau_add * w0_add
        scat_total = scat_base + scat_add

        w0_eff = np.zeros_like(tau_total)
        nonzero_tau = tau_total > 0
        w0_eff[nonzero_tau] = scat_total[nonzero_tau] / tau_total[nonzero_tau]

        g0_eff = np.zeros_like(tau_total)
        nonzero_scat = scat_total > 0
        g0_eff[nonzero_scat] = (
            scat_base[nonzero_scat] * g0_base[nonzero_scat]
            + scat_add[nonzero_scat] * g0_add[nonzero_scat]
        ) / scat_total[nonzero_scat]

        combined = base[["pressure", "wavenumber"]].copy()
        combined["opd"] = tau_total
        combined["w0"] = w0_eff
        combined["g0"] = g0_eff
        return combined

    def _run_case(cloud_df):
        case = deepcopy(earth)
        case.clouds(df=cloud_df, do_holes=False)
        return case.spectrum(opacity, calculation="reflected")

    df_clear = _run_case(haze_df)
    df_cloudy = _run_case(_combine_cloud_dfs(haze_df, water_cloud_df))

    df_res = deepcopy(df_clear)
    for col in df_res.columns:
        if col == "wavenumber":
            continue
        if col in df_cloudy.columns and np.issubdtype(df_res[col].dtype, np.number):
            df_res[col] = (1.0 - water_cloud_frac) * df_clear[col] + water_cloud_frac * df_cloudy[col]

    return df_res


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
        cloud_df=build_cloud_df(
            opacity,
            atm,
            cloud_scheme="rfast",
            cloud_top_pressure=10.0**log10_pc,
            cloud_thickness=10.0**log10_dpc,
            cloud_opd=10.0**log10_tauc,
            cloud_w0=0.99,
            cloud_g0=0.85,
        ),
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
