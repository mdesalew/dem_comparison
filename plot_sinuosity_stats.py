# Import packages
from joblib import Parallel, delayed

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import utils

# Read statistics and create DataFrame
country_codes = ['ESP', 'EST', 'ETH', 'USA']
dem_names = ['AW3D30', 'HydroSHEDS', 'MERIT', 'NASADEM', 'TanDEM']
df_list = []
for country_code in country_codes:
    for dem_name in dem_names:
        df = pd.read_csv(f'D:/dem_comparison/data/{country_code}/stats/{country_code}_{dem_name}_sinuosity.csv')
        df_list.append(df)
stats_df = pd.concat(df_list).reset_index(drop=True)
stats_df = stats_df.sort_values(['Country', 'DEM'])

# Calculate absolute sinuosity difference by DEM and write to CSV
grouped = stats_df.groupby('DEM')['Absolute difference'].describe().reset_index()
grouped = grouped.sort_values('mean').reset_index(drop=True)
grouped.to_csv(f'D:/dem_comparison/data/abs_sinuosity.csv', index=False)

# Create figure and save as PNG
color_names = ['magenta2', 'goldenrod2', 'seagreen3', 'tomato2', 'snow4']
sns.set_palette(sns.color_palette([utils.get_hex_code(color_name) for color_name in color_names]))
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
sns.barplot(y='Difference', x='Country', data=stats_df, hue='DEM', ax=ax)
ax.set_xticklabels([utils.get_basin_name(country_code) for country_code in country_codes])
ax.set_xlabel('Basin')
ax.set_ylabel('${S_{diff}}$ (%)')
for container in ax.containers:
    ax.bar_label(container)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, title='DEM')
plt.tight_layout()
plt.savefig(f'D:/dem_comparison/figures/sinuosity_by_country.png', dpi=300)
