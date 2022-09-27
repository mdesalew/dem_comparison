# Import packages
from joblib import Parallel, delayed

import utils

# Get input dictionary for forest statistics
country_codes = ['ESP', 'EST', 'ETH', 'USA']
dem_names = ['AW3D30', 'HydroSHEDS', 'MERIT', 'NASADEM', 'TanDEM']

# Get forest statistics in parallel
Parallel(n_jobs=4)(
    delayed(
        utils.get_sinuosity_stats
    )(country_code=country_code, dem_name=dem_name) for country_code in country_codes for dem_name in dem_names)
