# Import packages
import glob
from joblib import Parallel, delayed

import utils

# List of paths
paths = glob.glob(f'D:/dem_comparison/data/*/*_AW3D30.tif')

# Create slope raster from DEM in parallel
Parallel(n_jobs=4)(delayed(utils.create_slope_raster)(path) for path in paths)
