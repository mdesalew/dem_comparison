# Import packages
import glob
from joblib import Parallel, delayed

import utils

# List of paths
paths = [fp for fp in glob.glob(f'D:/dem_comparison/data/*/*/*.shp') if 'Ref' not in fp]

# Process points in parallel
Parallel(n_jobs=4)(delayed(utils.process_points)(path) for path in paths)
