# question : find common features across three files 

✅ Core Features to Standardize

For cross-mission ML modeling, the most useful features are:

Orbital Period (orbital_period)

Transit Duration (transit_duration)

Transit Depth (transit_depth)

Planet Radius (planet_radius)

Planet-Star Radius Ratio (radius_ratio)

Stellar Temperature (stellar_teff)

Stellar Radius (stellar_radius)

Stellar Mass (stellar_mass)

Insolation Flux (insolation_flux)

Equilibrium Temperature (teq)

Disposition (label)

# question : give me a mapping of these features across three files
schema_map = {
    'orbital_period': {'koi':'koi_period','toi':'pl_orbper','k2':'pl_orbper'},
    'transit_duration': {'koi':'koi_duration','toi':'pl_trandurh','k2':'pl_trandur'},
    'transit_depth': {'koi':'koi_depth','toi':'pl_trandep','k2':'pl_trandep'},
    'planet_radius': {'koi':'koi_prad','toi':'pl_rade','k2':'pl_rade'},
    'radius_ratio': {'koi':'koi_ror','toi':None,'k2':'pl_ratror'},
    'stellar_teff': {'koi':'koi_steff','toi':'st_teff','k2':'st_teff'},
    'stellar_radius': {'koi':'koi_srad','toi':'st_rad','k2':'st_rad'},
    'stellar_mass': {'koi':'koi_smass','toi':None,'k2':'st_mass'},
    'insolation_flux': {'koi':'koi_insol','toi':'pl_insol','k2':'pl_insol'},
    'teq': {'koi':'koi_teq','toi':'pl_eqt','k2':'pl_eqt'},
    'label': {'koi':'koi_disposition','toi':'tfopwg_disp','k2':'disposition'}
}

# question : using the mapping, give a code snippet to standardize the features in each file
