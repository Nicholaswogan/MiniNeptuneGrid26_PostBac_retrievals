import numpy as np
from scipy import constants as const
import numba as nb
from pathlib import Path
import pandas as pd
import miepython
from picaso import atmsetup, justdoit as jdi


_ATMSETUP_WEIGHT_HELPER = atmsetup.ATMSETUP.__new__(atmsetup.ATMSETUP)


@nb.njit()
def gravity(radius, mass, z):
    G_grav = const.G
    grav = G_grav * (mass/1.0e3) / ((radius + z)/1.0e2)**2.0
    grav = grav*1.0e2 # convert to cgs
    return grav

class McKayTitanHazeModel:
    """Steady-state Titan haze microphysics following McKay et al. (1989).

    The model assumes a single bulk haze population with one particle radius per
    altitude. The input production profile is a bulk haze mass source in
    ``g/cm^3/s`` on an altitude grid. The solver builds an internal uniformly
    spaced haze grid, solves the steady-state coagulation/sedimentation problem
    top-down, and interpolates the result back to the host grid.
    """

    def __init__(
            self,
            particle_density=1.0,
            charging_radius=0.09e-4,
            top_seed_radius=1.0e-7,
            haze_grid_size=600,
            top_extension_cm=0.0,
            sweep_clear_below_z=None,
            sweep_clear_below_pressure=None,
            viscosity_ref=1.76e-4,
            viscosity_ref_temperature=300.0,
            viscosity_temperature_exponent=0.7
        ):
        """Initialize the McKay Titan haze model.

        Parameters
        ----------
        particle_density : float, optional
            Bulk density of the haze material in g/cm^3.
        charging_radius : float, optional
            Charging radius in cm. The coagulation coefficient is reduced by a
            factor of ``exp(-r / charging_radius)``.
        top_seed_radius : float, optional
            Initial particle radius at the top of the haze grid in cm.
        haze_grid_size : int, optional
            Number of points in the internal uniformly spaced haze grid.
        top_extension_cm : float, optional
            Minimum height in cm by which the internal haze grid extends above
            the top of the supplied host grid.
        sweep_clear_below_z : float, optional
            If not ``None``, haze is forced to zero below this altitude in cm.
        sweep_clear_below_pressure : float, optional
            If not ``None``, haze is forced to zero at pressures larger than
            this value in dyn/cm^2.
        viscosity_ref : float, optional
            Reference dynamic viscosity in g/(cm s) used in the power-law
            viscosity parameterization.
        viscosity_ref_temperature : float, optional
            Reference temperature in K associated with ``viscosity_ref``.
        viscosity_temperature_exponent : float, optional
            Power-law exponent in the dynamic-viscosity relation
            ``viscosity_ref * (T / viscosity_ref_temperature)**exponent``.
        """

        self.particle_density = float(particle_density)
        self.charging_radius = float(charging_radius)
        self.top_seed_radius = float(top_seed_radius)
        self.haze_grid_size = int(haze_grid_size)
        self.top_extension_cm = float(top_extension_cm)
        self.sweep_clear_below_z = sweep_clear_below_z
        self.sweep_clear_below_pressure = sweep_clear_below_pressure
        self.viscosity_ref = float(viscosity_ref)
        self.viscosity_ref_temperature = float(viscosity_ref_temperature)
        self.viscosity_temperature_exponent = float(viscosity_temperature_exponent)

        if self.particle_density <= 0.0:
            raise ValueError("`particle_density` must be positive")
        if self.charging_radius <= 0.0:
            raise ValueError("`charging_radius` must be positive")
        if self.top_seed_radius <= 0.0:
            raise ValueError("`top_seed_radius` must be positive")
        if self.haze_grid_size < 4:
            raise ValueError("`haze_grid_size` must be at least 4")
        if self.top_extension_cm < 0.0:
            raise ValueError("`top_extension_cm` must be non-negative")

    @staticmethod
    def _require_1d(name, values):
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"`{name}` must be a 1D array")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"`{name}` contains non-finite values")
        return arr

    @staticmethod
    def _column_integral_from_top(z, q):
        out = np.zeros_like(q)
        for i in range(z.shape[0] - 2, -1, -1):
            dz = z[i + 1] - z[i]
            out[i] = out[i + 1] + 0.5*(q[i + 1] + q[i])*dz
        return out

    @staticmethod
    def _interp_to_grid(z_src, y_src, z_dst, *, log_values=False):
        if log_values:
            y_src = np.maximum(y_src, 1.0e-300)
            return 10.0**np.interp(z_dst, z_src, np.log10(y_src))
        return np.interp(z_dst, z_src, y_src)

    def _dynamic_viscosity(self, T):
        return self.viscosity_ref*(T/self.viscosity_ref_temperature)**self.viscosity_temperature_exponent

    @staticmethod
    def _pressure_to_cgs(pressure, pressure_unit):
        pressure = np.asarray(pressure, dtype=float)
        unit = pressure_unit.lower()
        if unit == 'bar':
            return pressure*1.0e6
        if unit in {'cgs', 'dyn/cm^2', 'dyn/cm2', 'barye'}:
            return pressure
        raise ValueError("`pressure_unit` must be 'bar' or 'dyn/cm^2'")

    @staticmethod
    def _number_density_from_state(pressure_dyn_cm2, temperature):
        k_boltz = const.Boltzmann*1.0e7
        return pressure_dyn_cm2/(k_boltz*temperature)

    @staticmethod
    def _surface_gravity(planet_radius, planet_mass):
        if planet_radius is None or planet_mass is None:
            raise ValueError("must provide both `planet_radius` and `planet_mass` when gravity is not supplied")
        return gravity(planet_radius, planet_mass, 0.0)

    def _gravity_profile(self, z, gravity_profile, planet_radius, planet_mass):
        if gravity_profile is not None:
            grav = self._require_1d('gravity_profile', gravity_profile)
            if grav.shape[0] != z.shape[0]:
                raise ValueError("`gravity_profile` must match the length of `z`")
            if np.any(grav <= 0.0):
                raise ValueError("`gravity_profile` must be positive")
            return grav

        if planet_radius is None or planet_mass is None:
            raise ValueError("must provide either `gravity_profile` or both `planet_radius` and `planet_mass`")

        return np.array([gravity(planet_radius, planet_mass, zi) for zi in z])

    def _integrate_hydrostatic_profile(self, pressure_dyn_cm2, temperature, mubar, grav):
        k_boltz = const.Boltzmann*1.0e7
        avogadro = const.Avogadro

        pressure_dyn_cm2 = self._require_1d('pressure', pressure_dyn_cm2)
        temperature = self._require_1d('temperature', temperature)
        mubar = self._require_1d('mubar', mubar)
        grav = self._require_1d('grav', grav)

        n = pressure_dyn_cm2.shape[0]
        for name, arr in [('temperature', temperature), ('mubar', mubar), ('grav', grav)]:
            if arr.shape[0] != n:
                raise ValueError(f"`{name}` must have length {n}")

        if np.any(pressure_dyn_cm2 <= 0.0):
            raise ValueError("`pressure` must be positive")
        if np.any(temperature <= 0.0):
            raise ValueError("`temperature` must be positive")
        if np.any(mubar <= 0.0):
            raise ValueError("`mubar` must be positive")
        if np.any(grav <= 0.0):
            raise ValueError("`grav` must be positive")

        z = np.zeros_like(pressure_dyn_cm2)
        for i in range(1, n):
            p_hi = pressure_dyn_cm2[i - 1]
            p_lo = pressure_dyn_cm2[i]
            if p_hi <= p_lo:
                raise ValueError("`pressure` must decrease monotonically with altitude")
            t_mid = 0.5*(temperature[i - 1] + temperature[i])
            mu_mid = 0.5*(mubar[i - 1] + mubar[i])
            g_mid = 0.5*(grav[i - 1] + grav[i])
            scale_height = k_boltz*t_mid/((mu_mid/avogadro)*g_mid)
            z[i] = z[i - 1] + scale_height*np.log(p_hi/p_lo)

        return z

    @staticmethod
    def _shift_profile_to_reference_pressure(pressure_dyn_cm2, z, reference_pressure_dyn_cm2):
        if reference_pressure_dyn_cm2 is None:
            return z
        pressure_dyn_cm2 = np.asarray(pressure_dyn_cm2, dtype=float)
        z = np.asarray(z, dtype=float)
        if pressure_dyn_cm2.shape != z.shape:
            raise ValueError("`pressure` and `z` must have the same shape")
        p_ref = float(reference_pressure_dyn_cm2)
        if p_ref <= 0.0:
            raise ValueError("`reference_pressure` must be positive")
        p_min = np.min(pressure_dyn_cm2)
        p_max = np.max(pressure_dyn_cm2)
        if not (p_min <= p_ref <= p_max):
            raise ValueError("`reference_pressure` must lie within the supplied pressure range")
        z_ref = np.interp(p_ref, pressure_dyn_cm2[::-1], z[::-1])
        return z - z_ref

    def _derive_z_and_gravity(
        self,
        pressure_dyn_cm2,
        temperature,
        mubar,
        *,
        gravity_profile=None,
        planet_radius=None,
        planet_mass=None,
        reference_pressure=None,
        n_iter=3,
    ):
        pressure_dyn_cm2 = self._require_1d('pressure', pressure_dyn_cm2)
        temperature = self._require_1d('temperature', temperature)
        mubar = self._require_1d('mubar', mubar)

        if gravity_profile is not None:
            grav = self._require_1d('gravity_profile', gravity_profile)
            if grav.shape[0] != pressure_dyn_cm2.shape[0]:
                raise ValueError("`gravity_profile` must have the same length as `pressure`")
            if np.any(grav <= 0.0):
                raise ValueError("`gravity_profile` must be positive")
            z = self._integrate_hydrostatic_profile(pressure_dyn_cm2, temperature, mubar, grav)
            z = self._shift_profile_to_reference_pressure(
                pressure_dyn_cm2,
                z,
                reference_pressure,
            )
            return z, grav

        g_surface = self._surface_gravity(planet_radius, planet_mass)
        grav_guess = np.full_like(pressure_dyn_cm2, g_surface)
        z = None
        for _ in range(n_iter):
            z = self._integrate_hydrostatic_profile(pressure_dyn_cm2, temperature, mubar, grav_guess)
            z = self._shift_profile_to_reference_pressure(pressure_dyn_cm2, z, reference_pressure)
            grav_guess = np.array([gravity(planet_radius, planet_mass, zi) for zi in z])

        return z, grav_guess

    def _prepare_host_state(
        self,
        pressure,
        temperature,
        q_mass,
        mubar,
        *,
        pressure_unit='dyn/cm^2',
        z=None,
        gravity_profile=None,
        planet_radius=None,
        planet_mass=None,
        reference_pressure=None,
    ):
        pressure = self._require_1d('pressure', pressure)
        temperature = self._require_1d('temperature', temperature)
        q_mass = self._require_1d('q_mass', q_mass)
        mubar = self._require_1d('mubar', mubar)

        n = pressure.shape[0]
        for name, arr in [('temperature', temperature), ('q_mass', q_mass), ('mubar', mubar)]:
            if arr.shape[0] != n:
                raise ValueError(f"`{name}` must have length {n}")

        if np.any(pressure <= 0.0):
            raise ValueError("`pressure` must be positive")
        if np.any(temperature <= 0.0):
            raise ValueError("`temperature` must be positive")
        if np.any(q_mass < 0.0):
            raise ValueError("`q_mass` must be non-negative")
        if np.any(mubar <= 0.0):
            raise ValueError("`mubar` must be positive")

        pressure_dyn_cm2 = self._pressure_to_cgs(pressure, pressure_unit)

        if np.all(np.diff(pressure_dyn_cm2) > 0.0):
            if z is not None:
                raise ValueError(
                    "`z` should not be supplied when `pressure` is ordered top-to-bottom; "
                    "use `solve_from_atmosphere` or pre-order the profiles bottom-to-top."
                )
            pressure_dyn_cm2 = pressure_dyn_cm2[::-1].copy()
            temperature = temperature[::-1].copy()
            q_mass = q_mass[::-1].copy()
            mubar = mubar[::-1].copy()
            if gravity_profile is not None:
                gravity_profile = self._require_1d('gravity_profile', gravity_profile)[::-1].copy()
        elif not np.all(np.diff(pressure_dyn_cm2) < 0.0):
            raise ValueError("`pressure` must be strictly monotonic")
        else:
            if z is not None:
                z = self._require_1d('z', z)
            if gravity_profile is not None:
                gravity_profile = self._require_1d('gravity_profile', gravity_profile)

        if z is None:
            if gravity_profile is None:
                z, gravity_profile = self._derive_z_and_gravity(
                    pressure_dyn_cm2,
                    temperature,
                    mubar,
                    planet_radius=planet_radius,
                    planet_mass=planet_mass,
                    reference_pressure=reference_pressure,
                )
            else:
                z, gravity_profile = self._derive_z_and_gravity(
                    pressure_dyn_cm2,
                    temperature,
                    mubar,
                    gravity_profile=gravity_profile,
                    reference_pressure=reference_pressure,
                )
        else:
            if z.shape[0] != n:
                raise ValueError("`z` must have the same length as `pressure`")
            if not (np.all(np.diff(z) > 0.0) or np.all(np.diff(z) < 0.0)):
                raise ValueError("`z` must be strictly monotonic")
            gravity_profile = self._gravity_profile(z, gravity_profile, planet_radius, planet_mass)
            if np.all(np.diff(z) < 0.0):
                raise ValueError("`z` must increase with altitude to match PICASO-style profiles")

        n_atm = self._number_density_from_state(pressure_dyn_cm2, temperature)

        return {
            'z': z,
            'P': pressure_dyn_cm2,
            'T': temperature,
            'n_atm': n_atm,
            'mubar': mubar,
            'q_mass': q_mass,
            'grav': gravity_profile
        }

    def _build_haze_grid(self, host):
        z = host['z']
        T = host['T']
        grav = host['grav']
        mubar = host['mubar']

        k_boltz = const.Boltzmann*1.0e7
        m_particle = mubar[-1]/const.Avogadro
        scale_height_top = k_boltz*T[-1]/(m_particle*grav[-1])
        z_top = z[-1] + max(self.top_extension_cm, 3.0*scale_height_top)
        z_haze = np.linspace(z[0], z_top, self.haze_grid_size)

        P_haze = np.empty_like(z_haze)
        T_haze = np.empty_like(z_haze)
        n_atm_haze = np.empty_like(z_haze)
        mubar_haze = np.empty_like(z_haze)
        grav_haze = np.empty_like(z_haze)
        q_haze = np.empty_like(z_haze)

        inside = z_haze <= z[-1]
        outside = ~inside

        P_haze[inside] = self._interp_to_grid(z, host['P'], z_haze[inside], log_values=True)
        T_haze[inside] = self._interp_to_grid(z, host['T'], z_haze[inside])
        n_atm_haze[inside] = self._interp_to_grid(z, host['n_atm'], z_haze[inside], log_values=True)
        mubar_haze[inside] = self._interp_to_grid(z, host['mubar'], z_haze[inside])
        grav_haze[inside] = self._interp_to_grid(z, host['grav'], z_haze[inside])
        q_haze[inside] = self._interp_to_grid(z, host['q_mass'], z_haze[inside])

        if np.any(outside):
            z_extra = z_haze[outside]
            dz = z_extra - z[-1]
            T_haze[outside] = host['T'][-1]
            mubar_haze[outside] = host['mubar'][-1]
            grav_haze[outside] = host['grav'][-1]*(z[-1] + 1.0)**2/(z_extra + 1.0)**2

            scale_height = k_boltz*host['T'][-1]/((host['mubar'][-1]/const.Avogadro)*host['grav'][-1])
            P_haze[outside] = host['P'][-1]*np.exp(-dz/scale_height)
            n_atm_haze[outside] = host['n_atm'][-1]*np.exp(-dz/scale_height)
            q_haze[outside] = 0.0

        return {
            'z': z_haze,
            'P': P_haze,
            'T': T_haze,
            'n_atm': n_atm_haze,
            'mubar': mubar_haze,
            'grav': grav_haze,
            'q_mass': q_haze
        }

    def _transport_coefficients(self, r, T, n_atm, mubar, grav):
        k_boltz = const.Boltzmann*1.0e7
        m_gas = mubar/const.Avogadro
        rho_gas = n_atm*m_gas
        viscosity = self._dynamic_viscosity(T)
        thermal_speed = np.sqrt(8.0*k_boltz*T/(np.pi*m_gas))
        mean_free_path = 3.0*viscosity/(rho_gas*thermal_speed)

        v_free = self.particle_density*grav*np.sqrt(np.pi*m_gas/(8.0*k_boltz*T))*r/(n_atm*m_gas)
        v_stokes = 2.0*self.particle_density*grav*r**2/(9.0*viscosity)

        kn = mean_free_path/np.maximum(r, 1.0e-300)
        weight = kn/(1.0 + kn)
        v_settle = weight*v_free + (1.0 - weight)*v_stokes

        K_free = 4.0*np.sqrt(3.0*k_boltz*T/self.particle_density)*np.sqrt(r)
        K_stokes = 8.0*k_boltz*T/(3.0*viscosity)
        K_coag = weight*K_free + (1.0 - weight)*K_stokes
        K_coag *= np.exp(-r/self.charging_radius)

        return {
            'v_settle': v_settle,
            'K_coag': K_coag,
            'mean_free_path': mean_free_path,
            'viscosity': viscosity,
            'knudsen': kn
        }

    def _sweep_clear_mask(self, z, P):
        mask = np.zeros_like(z, dtype=bool)
        if self.sweep_clear_below_z is not None:
            mask |= z < self.sweep_clear_below_z
        if self.sweep_clear_below_pressure is not None:
            mask |= P > self.sweep_clear_below_pressure
        return mask

    def gaussian_production_profile(self, z, P, column_production, peak_pressure, width_pressure):
        """Construct an approximate Gaussian haze source in pressure.

        Parameters
        ----------
        z : ndarray
            Altitude in cm.
        P : ndarray
            Pressure in dyn/cm^2.
        column_production : float
            Total integrated haze mass production in g/cm^2/s.
        peak_pressure : float
            Pressure of peak production in dyn/cm^2.
        width_pressure : float
            Gaussian width in dyn/cm^2.

        Returns
        -------
        ndarray
            Haze mass source in g/cm^3/s.
        """
        z = self._require_1d('z', z)
        P = self._require_1d('P', P)
        if z.shape[0] != P.shape[0]:
            raise ValueError("`z` and `P` must have the same length")
        if column_production < 0.0:
            raise ValueError("`column_production` must be non-negative")
        if peak_pressure <= 0.0 or width_pressure <= 0.0:
            raise ValueError("`peak_pressure` and `width_pressure` must be positive")

        shape = np.exp(-((P - peak_pressure)/width_pressure)**2)
        norm = np.trapz(shape, z)
        if norm <= 0.0:
            return np.zeros_like(z)
        return column_production*shape/norm

    def solve(
        self,
        pressure,
        temperature,
        q_mass,
        *,
        mubar,
        pressure_unit='dyn/cm^2',
        z=None,
        gravity_profile=None,
        planet_radius=None,
        planet_mass=None,
        reference_pressure=None,
    ):
        """Solve the steady-state McKay haze model.

        Parameters
        ----------
        pressure : ndarray
            Pressure profile.
        temperature : ndarray
            Temperature profile in K.
        q_mass : ndarray
            Bulk haze mass source in g/cm^3/s.
        mubar : ndarray
            Mean molecular weight profile in g/mol.
        pressure_unit : str, optional
            Units of ``pressure``. Use ``'bar'`` for PICASO-style inputs.
        z : ndarray, optional
            Altitude profile in cm. If omitted, the profile is derived from
            hydrostatic balance.
        gravity_profile : ndarray, optional
            Gravity profile in cm/s^2. If omitted, and ``planet_radius`` and
            ``planet_mass`` are provided, gravity is computed internally.
        planet_radius : float, optional
            Planet radius in cm if gravity must be derived.
        planet_mass : float, optional
            Planet mass in g if gravity must be derived.
        reference_pressure : float, optional
            Pressure in the same units as ``pressure`` that corresponds to
            ``planet_radius``. Used to anchor the altitude zero-point for gas
            giants.

        Returns
        -------
        dict
            Haze properties on the internal haze grid and interpolated host grid.
        """
        host = self._prepare_host_state(
            pressure,
            temperature,
            q_mass,
            mubar,
            pressure_unit=pressure_unit,
            z=z,
            gravity_profile=gravity_profile,
            planet_radius=planet_radius,
            planet_mass=planet_mass,
            reference_pressure=reference_pressure,
        )
        haze = self._build_haze_grid(host)

        z_haze = haze['z']
        C_haze = self._column_integral_from_top(z_haze, haze['q_mass'])

        r_haze = np.zeros_like(z_haze)
        n_haze = np.zeros_like(z_haze)
        haze_mass_density = np.zeros_like(z_haze)
        v_settle = np.zeros_like(z_haze)
        K_coag = np.zeros_like(z_haze)
        mean_free_path = np.zeros_like(z_haze)
        knudsen = np.zeros_like(z_haze)

        A = (4.0/3.0)*np.pi*self.particle_density
        r_haze[-1] = self.top_seed_radius

        for i in range(z_haze.shape[0] - 1, -1, -1):
            r_now = max(r_haze[i], self.top_seed_radius)
            coeffs = self._transport_coefficients(r_now, haze['T'][i], haze['n_atm'][i], haze['mubar'][i], haze['grav'][i])

            v_mag = max(coeffs['v_settle'], 1.0e-30)
            v_settle[i] = coeffs['v_settle']
            K_coag[i] = coeffs['K_coag']
            mean_free_path[i] = coeffs['mean_free_path']
            knudsen[i] = coeffs['knudsen']

            if C_haze[i] > 0.0:
                n_haze[i] = C_haze[i]/(A*r_now**3*v_mag)
                haze_mass_density[i] = n_haze[i]*A*r_now**3

            if i == 0:
                continue

            dz = z_haze[i - 1] - z_haze[i]
            if C_haze[i] <= 0.0:
                r_haze[i - 1] = r_now
                continue

            dr_dz = -(coeffs['K_coag']*C_haze[i])/(3.0*A*r_now**2*v_mag**2)
            dr_dz -= (haze['q_mass'][i]*r_now)/(3.0*C_haze[i])

            r_next = r_now + dr_dz*dz
            r_haze[i - 1] = max(r_next, self.top_seed_radius)

        sweep_mask_haze = self._sweep_clear_mask(haze['z'], haze['P'])
        r_haze[sweep_mask_haze] = self.top_seed_radius
        n_haze[sweep_mask_haze] = 0.0
        haze_mass_density[sweep_mask_haze] = 0.0

        coeffs_final = self._transport_coefficients(r_haze, haze['T'], haze['n_atm'], haze['mubar'], haze['grav'])
        v_settle = coeffs_final['v_settle']
        K_coag = coeffs_final['K_coag']
        mean_free_path = coeffs_final['mean_free_path']
        knudsen = coeffs_final['knudsen']

        host_r = self._interp_to_grid(haze['z'], r_haze, host['z'], log_values=True)
        host_n = self._interp_to_grid(haze['z'], n_haze, host['z'], log_values=True)
        host_mass = self._interp_to_grid(haze['z'], haze_mass_density, host['z'], log_values=True)
        host_v = self._interp_to_grid(haze['z'], v_settle, host['z'], log_values=True)
        host_C = self._interp_to_grid(haze['z'], C_haze, host['z'], log_values=True)
        host_q = self._interp_to_grid(haze['z'], haze['q_mass'], host['z'])

        sweep_mask_host = self._sweep_clear_mask(host['z'], host['P'])
        host_n[sweep_mask_host] = 0.0
        host_mass[sweep_mask_host] = 0.0
        host_r[sweep_mask_host] = self.top_seed_radius

        return {
            'haze_grid': {
                'z': haze['z'],
                'P': haze['P'],
                'T': haze['T'],
                'n_atm': haze['n_atm'],
                'mubar': haze['mubar'],
                'q_mass': haze['q_mass'],
                'column_production': C_haze,
                'radius': r_haze,
                'number_density': n_haze,
                'mass_density': haze_mass_density,
                'settling_velocity': v_settle,
                'coagulation_coefficient': K_coag,
                'mean_free_path': mean_free_path,
                'knudsen': knudsen
            },
            'host_grid': {
                'z': host['z'],
                'P': host['P'],
                'T': host['T'],
                'n_atm': host['n_atm'],
                'mubar': host['mubar'],
                'q_mass': host_q,
                'column_production': host_C,
                'radius': host_r,
                'number_density': host_n,
                'mass_density': host_mass,
                'settling_velocity': host_v
            }
        }

    def solve_from_atmosphere(
        self,
        atmosphere,
        q_mass=None,
        *,
        mubar=None,
        column_production=None,
        peak_pressure=None,
        width_pressure=None,
        reference_pressure=None,
        gravity_profile=None,
        planet_radius=None,
        planet_mass=None,
    ):
        """Convenience wrapper for PICASO-like atmosphere inputs.

        Parameters
        ----------
        atmosphere : pandas.DataFrame or dict
            Atmosphere container with pressure and temperature columns.
        q_mass : ndarray
            Bulk haze source profile on the same vertical grid.
        column_production : float, optional
            Total integrated haze production in g/cm^2/s used to build a
            Gaussian source profile if ``q_mass`` is not supplied.
        peak_pressure : float, optional
            Pressure of peak haze production in bar, used with
            ``column_production`` and ``width_pressure``.
        width_pressure : float, optional
            Gaussian pressure width in bar, used with ``column_production`` and
            ``peak_pressure``.
        mubar : ndarray, optional
            Mean molecular weight profile. If omitted, it is derived from the
            composition columns in ``atmosphere``.
        gravity_profile, planet_radius, planet_mass : optional
            Passed through to :meth:`solve`.
        reference_pressure : float, optional
            Reference pressure in bar at which ``planet_radius`` is defined for
            gas giants. If omitted, the deepest supplied pressure is treated as
            the reference level.
        planet_radius : float, optional
            Planet radius in Earth radii. Converted internally to cm for the
            haze solver.
        planet_mass : float, optional
            Planet mass in Earth masses. Converted internally to g for the
            haze solver.
        """
        if not isinstance(atmosphere, pd.DataFrame):
            raise TypeError("`atmosphere` must be a pandas DataFrame")
        if 'pressure' not in atmosphere.columns:
            raise ValueError("missing required atmosphere column 'pressure'")
        if 'temperature' not in atmosphere.columns:
            raise ValueError("missing required atmosphere column 'temperature'")
        pressure = atmosphere['pressure'].to_numpy(dtype=float)
        temperature = atmosphere['temperature'].to_numpy(dtype=float)
        if mubar is None:
            species_cols = []
            mol_weights = []
            x_cols = []
            for c in atmosphere.columns:
                if c in {'pressure', 'temperature'}:
                    continue
                try:
                    weights = _ATMSETUP_WEIGHT_HELPER.get_weights([c])
                    mol_weights.append(float(weights[c]))
                    species_cols.append(c)
                    x_cols.append(atmosphere[c].to_numpy(dtype=float))
                except Exception:
                    continue
            if len(species_cols) == 0:
                raise ValueError("no PICASO-style composition columns available to derive `mubar`")
            mol_weights = np.asarray(mol_weights, dtype=float)
            x = np.column_stack(x_cols)
            mubar = np.sum(x*mol_weights[None, :], axis=1)

        if mubar is None:
            raise ValueError("`mubar` must be provided or derivable from the atmosphere")

        pressure_cgs = pressure*1.0e6
        reference_pressure_cgs = None if reference_pressure is None else reference_pressure*1.0e6
        planet_radius_cgs = None if planet_radius is None else planet_radius*6.371e8
        planet_mass_cgs = None if planet_mass is None else planet_mass*5.9722e27

        if q_mass is None:
            if column_production is None or peak_pressure is None or width_pressure is None:
                raise ValueError(
                    "provide either `q_mass` or all of `column_production`, "
                    "`peak_pressure`, and `width_pressure`"
                )
            z_source, _ = self._derive_z_and_gravity(
                pressure_cgs[::-1].copy(),
                temperature[::-1].copy(),
                mubar[::-1].copy(),
                gravity_profile=gravity_profile[::-1].copy() if gravity_profile is not None else None,
                planet_radius=planet_radius_cgs,
                planet_mass=planet_mass_cgs,
                reference_pressure=reference_pressure_cgs,
            )
            pressure_source = pressure_cgs[::-1].copy()
            q_mass_source = self.gaussian_production_profile(
                z_source,
                pressure_source,
                column_production,
                peak_pressure*1.0e6,
                width_pressure*1.0e6,
            )
            q_mass = q_mass_source[::-1].copy()

        return self.solve(
            pressure_cgs,
            temperature,
            q_mass,
            mubar=mubar,
            reference_pressure=reference_pressure_cgs,
            gravity_profile=gravity_profile,
            planet_radius=planet_radius_cgs,
            planet_mass=planet_mass_cgs,
        )
    

def _miepython_optics_cache(
        refrind_path,
        x_v,
        x_i,
        solar_cutoff_microns,
        wave_grid_microns,
        radii_cm
    ):

    wave_in, nn, kk = _read_refrind_file(refrind_path)

    wave_grid = np.asarray(wave_grid_microns, dtype=float)
    if wave_grid.ndim != 1:
        raise ValueError("`wave_grid_microns` must be a 1D array")
    if wave_grid.shape[0] < 2:
        raise ValueError("`wave_grid_microns` must contain at least two points")
    if not np.all(np.isfinite(wave_grid)):
        raise ValueError("`wave_grid_microns` contains non-finite values")
    if np.any(wave_grid <= 0.0):
        raise ValueError("`wave_grid_microns` must be positive")

    nn_grid = np.interp(wave_grid, wave_in, nn)
    kk_grid = np.interp(wave_grid, wave_in, kk)
    kk_grid = kk_grid*np.where(wave_grid <= solar_cutoff_microns, x_v, x_i)

    radius_grid = np.asarray(radii_cm, dtype=float)
    n_radii = radius_grid.shape[0]
    wave_nm = wave_grid*1.0e3
    qext = np.empty((wave_grid.shape[0], n_radii))
    qscat = np.empty((wave_grid.shape[0], n_radii))
    cos_qscat = np.empty((wave_grid.shape[0], n_radii))

    for iw, wavelength_nm in enumerate(wave_nm):
        m_eff = nn_grid[iw] - (1j)*kk_grid[iw]
        for ir, radius_cm in enumerate(radius_grid):
            size_parameter = 2.0*np.pi*(radius_cm*1.0e7)/wavelength_nm
            qext_i, qsca_i, _, g_i = miepython.efficiencies_mx(m_eff, size_parameter)
            qext[iw, ir] = qext_i
            qscat[iw, ir] = qsca_i
            cos_qscat[iw, ir] = qsca_i*g_i

    return wave_grid, radius_grid, qext, qscat, cos_qscat


def _get_picaso_cloud_wavenumber_grid():
    return np.asarray(jdi.get_cld_input_grid("wave_EGP.dat"), dtype=float)


def _coerce_haze_profiles(values, name, nlayer):
    if isinstance(values, dict):
        out = {}
        for key, val in values.items():
            arr = np.asarray(val, dtype=float)
            if arr.ndim != 1:
                raise ValueError(f"`{name}` for species '{key}' must be a 1D array")
            if arr.shape[0] != nlayer:
                raise ValueError(
                    f"`{name}` for species '{key}' has length {arr.shape[0]}, expected {nlayer}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"`{name}` for species '{key}' contains non-finite values")
            out[key] = arr
        return out

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"`{name}` must be a 1D array or a dict of 1D arrays")
    if arr.shape[0] != nlayer:
        raise ValueError(f"`{name}` has length {arr.shape[0]}, expected {nlayer}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"`{name}` contains non-finite values")
    return {'haze': arr}


def _validate_refrind_file(refractive_index_file):
    path = Path(refractive_index_file)
    if path.suffix != '.refrind':
        raise ValueError("`refractive_index_file` must be a VIRGA-style `.refrind` file")
    if not path.exists():
        raise FileNotFoundError(path)
    return path.resolve()


def _read_refrind_file(refrind_path):
    data = np.loadtxt(refrind_path, usecols=[1, 2, 3])
    wave_in = np.asarray(data[:, 0], dtype=float)
    nn = np.asarray(data[:, 1], dtype=float)
    kk = np.asarray(data[:, 2], dtype=float)

    if wave_in[0] > wave_in[-1]:
        wave_in = wave_in[::-1].copy()
        nn = nn[::-1].copy()
        kk = kk[::-1].copy()

    return wave_in, nn, kk


def make_picaso_haze_clouddf(
        pressure,
        dz,
        densities,
        radii,
        *,
        refractive_index_file='input/khare_tholins.refrind',
        x_v=4/3,
        x_i=0.5,
        solar_cutoff_microns=5.0,
    ):
    """Build a PICASO cloud DataFrame from haze profiles and VIRGA Mie optics.

    Parameters
    ----------
    pressure : ndarray of shape (nlayer,)
        Layer pressures in bar.
    dz : ndarray of shape (nlayer,)
        Layer thicknesses in cm.
    densities : ndarray of shape (nlayer,) or dict[str, ndarray]
        Particle number densities in cm^-3. If a dict is passed, each key is
        treated as a separate haze species and all species are summed.
    radii : ndarray of shape (nlayer,) or dict[str, ndarray]
        Particle radii in cm. If a dict is passed, its keys must match
        ``densities``.
    refractive_index_file : str or path-like, optional
        Path to a VIRGA-style ``.refrind`` file containing four numeric
        columns: index, wavelength in microns, real refractive index, and
        imaginary refractive index.
    x_v : float, optional
        McKay-style visible/solar haze absorption scaling factor. This is
        applied to the imaginary refractive index at wavelengths shorter than
        or equal to ``solar_cutoff_microns``.
    x_i : float, optional
        McKay-style thermal-infrared haze absorption scaling factor. This is
        applied to the imaginary refractive index at wavelengths longer than
        ``solar_cutoff_microns``.
    solar_cutoff_microns : float, optional
        Wavelength boundary, in microns, between the McKay ``Xv`` and ``Xi``
        regimes. The default of 5 microns follows the paper's separation
        between solar/near-infrared and thermal-infrared calculations.

    Returns
    -------
    pandas.DataFrame
        PICASO-ready cloud table with columns ``pressure``, ``wavenumber``,
        ``opd``, ``w0``, and ``g0``. The table contains one row per
        layer-wavelength pair and is sorted by ascending pressure and
        wavenumber.

    Notes
    -----
    The optical data are interpolated in ``log10(radius)`` space with radii
    clamped to the tabulated range. The extinction optical depth is computed
    as ``opd = qext * pi * r^2 * n * dz``.

    McKay et al. (1989) scale the haze imaginary refractive index relative to
    Khare et al. (1984) with separate visible and infrared factors ``Xv`` and
    ``Xi``. This helper applies those factors directly to the imaginary
    refractive index before calling the selected Mie backend.
    """

    pressure = np.asarray(pressure, dtype=float)
    dz = np.asarray(dz, dtype=float)
    if pressure.ndim != 1:
        raise ValueError("`pressure` must be a 1D array")
    if dz.ndim != 1:
        raise ValueError("`dz` must be a 1D array")
    if pressure.shape[0] != dz.shape[0]:
        raise ValueError("`pressure` and `dz` must have the same length")
    if not np.all(np.isfinite(pressure)):
        raise ValueError("`pressure` contains non-finite values")
    if not np.all(np.isfinite(dz)):
        raise ValueError("`dz` contains non-finite values")
    if x_v <= 0.0 or x_i <= 0.0:
        raise ValueError("`x_v` and `x_i` must be positive")
    if solar_cutoff_microns <= 0.0:
        raise ValueError("`solar_cutoff_microns` must be positive")

    nlayer = pressure.shape[0]
    cloud_wavenumber = _get_picaso_cloud_wavenumber_grid()
    if cloud_wavenumber.ndim != 1:
        raise ValueError("PICASO cloud grid must be a 1D array")
    if cloud_wavenumber.shape[0] < 2:
        raise ValueError("PICASO cloud grid must contain at least two points")
    if not np.all(np.isfinite(cloud_wavenumber)):
        raise ValueError("PICASO cloud grid contains non-finite values")
    if np.any(cloud_wavenumber <= 0.0):
        raise ValueError("PICASO cloud grid must be positive")

    density_profiles = _coerce_haze_profiles(densities, 'densities', nlayer)
    radius_profiles = _coerce_haze_profiles(radii, 'radii', nlayer)
    if set(density_profiles) != set(radius_profiles):
        raise ValueError("`densities` and `radii` must have identical species keys")
    refrind_path = _validate_refrind_file(refractive_index_file)

    radii_all = np.concatenate([radius_profiles[key] for key in sorted(radius_profiles)])
    if np.any(radii_all <= 0.0):
        raise ValueError("all `radii` values must be positive")
    radii_cm_grid = np.unique(radii_all.astype(float))
    wave_grid_microns = 1.0e4 / cloud_wavenumber
    wavelengths_microns, radii_cm_grid, qext, qsca, cos_qscat = _miepython_optics_cache(
        str(refrind_path),
        float(x_v),
        float(x_i),
        float(solar_cutoff_microns),
        wave_grid_microns,
        tuple(radii_cm_grid.tolist())
    )

    nwave = wavelengths_microns.shape[0]
    w0 = np.divide(qsca, qext, out=np.zeros_like(qext), where=qext > 0.0)
    g0 = np.divide(cos_qscat, qsca, out=np.zeros_like(qsca), where=qsca > 0.0)

    tau_total = np.zeros((nlayer, nwave))
    tau_w0_total = np.zeros((nlayer, nwave))
    tau_scat_g0_total = np.zeros((nlayer, nwave))

    for key in density_profiles:
        density = density_profiles[key]
        radius_cm = radius_profiles[key]
        if np.any(radius_cm <= 0.0):
            raise ValueError(f"`radii` for species '{key}' must be positive")
        if np.any(density < 0.0):
            raise ValueError(f"`densities` for species '{key}' must be non-negative")

        radius_index = np.searchsorted(radii_cm_grid, radius_cm)
        if not np.array_equal(radii_cm_grid[radius_index], radius_cm):
            raise ValueError("`miepython` backend expected exact input radii on its optics grid")
        qext_layer = qext[:, radius_index].T
        w0_layer = w0[:, radius_index].T
        g0_layer = g0[:, radius_index].T

        sigma_ext = qext_layer*(np.pi*radius_cm[:, None]**2)
        tau = sigma_ext*density[:, None]*dz[:, None]
        tau_scat = tau*w0_layer

        tau_total += tau
        tau_w0_total += tau_scat
        tau_scat_g0_total += tau_scat*g0_layer

    w0_mix = np.divide(
        tau_w0_total,
        tau_total,
        out=np.zeros_like(tau_total),
        where=tau_total > 0.0
    )
    g0_mix = np.divide(
        tau_scat_g0_total,
        tau_w0_total,
        out=np.zeros_like(tau_total),
        where=tau_w0_total > 0.0
    )

    pressure_all = np.repeat(pressure, nwave)
    wavenumber_all = np.tile(cloud_wavenumber, nlayer)
    cloud_df = pd.DataFrame({
        'pressure': pressure_all,
        'wavenumber': wavenumber_all,
        'opd': tau_total.reshape(-1),
        'w0': w0_mix.reshape(-1),
        'g0': g0_mix.reshape(-1)
    })
    cloud_df = cloud_df.sort_values(['pressure', 'wavenumber']).reset_index(drop=True)

    return cloud_df


def make_picaso_haze_clouddf_from_solution(
        solution,
        *,
        refractive_index_file='input/khare_tholins.refrind',
        x_v=4/3,
        x_i=0.5,
        solar_cutoff_microns=5.0,
    ):
    """Build a PICASO haze dataframe directly from :meth:`solve` output.

    Parameters
    ----------
    solution : dict
        Output of :meth:`McKayTitanHazeModel.solve` or its ``host_grid`` entry.
    Other parameters
        Passed through to :func:`make_picaso_haze_clouddf`.
    """
    host = solution.get('host_grid', solution)
    if not isinstance(host, dict):
        raise TypeError("`solution` must be the dict returned by `solve` or `solve()['host_grid']`")

    pressure = np.asarray(host['P'], dtype=float)
    z = np.asarray(host['z'], dtype=float)
    densities = np.asarray(host['number_density'], dtype=float)
    radii = np.asarray(host['radius'], dtype=float)

    if pressure.shape[0] < 2:
        raise ValueError("solution must contain at least two vertical levels")
    if not (z.shape == pressure.shape == densities.shape == radii.shape):
        raise ValueError("solution host-grid arrays must all have the same length")

    # PICASO expects one cloud row per atmospheric layer, not per level.
    pressure_bar = np.sqrt(pressure[:-1] * pressure[1:]) / 1.0e6
    dz = np.abs(np.diff(z))
    densities_layer = 0.5*(densities[:-1] + densities[1:])
    radii_layer = 0.5*(radii[:-1] + radii[1:])

    return make_picaso_haze_clouddf(
        pressure_bar,
        dz,
        densities_layer,
        radii_layer,
        refractive_index_file=refractive_index_file,
        x_v=x_v,
        x_i=x_i,
        solar_cutoff_microns=solar_cutoff_microns,
    )
