# Elizabeth's notes

Case 3: H2-He Rich Mini Neptune around a Sun-Like Star (that matched with Archean Earth):

Planet:

Input Parameters:
- planet radius: 2x Earth Radius
- metallicity: 2x (logspace) x solar (~100x solar metallicity)
- tint: 50K
- semi major: 1 AU
- ctoO_solar: 0.7525 x solar c/o ratio
- kzz: 5 cm^2/s

Calculated Parameters:
- planet mass: 4.657280767574196x Earth Mass (from chen_kipping_2017 mass/radius relation)
- planet Teq: 278.6545412784493 K
- planet gravity: 11.433593 m / (s2)

Star:
-stellar radius = 1x Solar
-stellar Teff = 5778 K
-stellar metallicity = 0 (log10 of stellar)
-stellar logg = 4.4 (log10(gravity) in cgs units)

Photochem Profile (pressure and temperature are saved which pulled from PICASSO PT Profile): PhotochemPT_MiniNep_2.0_2.0_50.0_1.0_0.7525_5.0.pkl

Planet:
Phase = 0 degrees/radians (full phase)

Cloud Parameters?
W0 = 0.99 (single scattering albedo)
G0 = 0.85 (asymmetry factor)
Opd (optical depth per layer is constant)  = 10 (total extinction of each layer)
Cloud Top Pressure (Pa) (in Salvador paper, p reported in log10 bars) = 0.0348 bars
Cloud Thickness (Pa) (in Salvador paper, dp reported in log10 bars) = 3.0334 bars
Cloudiness Fraction = 0.5

w/ opacity at R=15,000 (binning at 150), [0.1 - 2.5 microns] → RLS_H2HeRich_MiniNep_R15000_cld0.5.pkl

w/ opacity at R=60,000 (binning at 3000), [0.1 - 2.5 microns] →  TBD, have some minor adjustments to my HYAK environment (my computer will not run it).



<!-- ln -s /Users/nicholas/Applications/picaso_data/reference/opacities/opacities_photochem_0.1_250.0_R15000_v2.db picasofiles/opacities_photochem_0.1_250.0_R15000_v2.db -->