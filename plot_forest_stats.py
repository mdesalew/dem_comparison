# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import utils

stat_name = 'forest_pct'
class_func = utils.classify_forest_pct
stat_name_long = 'Forest percentage'
palette = 'light:seagreen'

country_codes = ['ESP', 'EST', 'ETH', 'USA']
dem_names = ['AW3D30', 'HydroSHEDS', 'MERIT', 'NASADEM', 'TanDEM']
feature_types = ['basin', 'stream']

# Plot relationship between distance to reference and statistic
for feature_type in feature_types:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for country_code, ax in zip(country_codes, axes.flatten()):
        dist_stats = pd.\
            concat(
            [pd.read_csv(utils.get_fp_to_stats(country_code, dem_name, feature_type, 'dist_to_ref')) for dem_name in
                    dem_names]).reset_index(drop=True)
        stats = pd.\
            concat(
            [pd.read_csv(utils.get_fp_to_stats(country_code, dem_name, feature_type, stat_name)) for dem_name in
                    dem_names]).reset_index(drop=True)
        merged = dist_stats.merge(stats[['point_id', stat_name]], how='left', on='point_id')
        merged = merged.sort_values(['dem_name', stat_name]).reset_index(drop=True)
        merged[f'{stat_name}_class'] = merged.apply(class_func, axis=1)
        sns.boxplot(y='dist_to_ref', x=f'{stat_name}_class', data=merged, hue='dem_name', ax=ax, showfliers=False)
        ax.set_xlabel(stat_name_long)
        ax.set_ylabel(f'Distance to reference {feature_type}')
        ax.set_title(country_code)
        if country_code != country_codes[-1]:
            ax.get_legend().remove()
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    fig.legend(handles, labels, loc='lower center', ncol=5, title='DEM', bbox_to_anchor=(0.5, -0.1))
    for ax in axes.flatten():
        ax.set(ylim=(-50, 1300))
    plt.tight_layout()
    plt.savefig(
        f'D:/dem_comparison/figures/{stat_name}_vs_dist_to_ref_{feature_type}.png', dpi=300, bbox_inches='tight'
    )

# Plot relationship between distance to reference and statistic by class
for feature_type in feature_types:
    df_list = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for country_code, ax in zip(country_codes, axes.flatten()):
        dist_stats = pd.\
            concat(
            [pd.read_csv(utils.get_fp_to_stats(country_code, dem_name, feature_type, 'dist_to_ref')) for dem_name in
                    dem_names]).reset_index(drop=True)
        stats = pd.\
            concat(
            [pd.read_csv(utils.get_fp_to_stats(country_code, dem_name, feature_type, stat_name)) for dem_name in
                    dem_names]).reset_index(drop=True)
        merged = dist_stats.merge(stats[['point_id', stat_name]], how='left', on='point_id')
        merged = merged.sort_values(['dem_name', stat_name]).reset_index(drop=True)
        merged[f'{stat_name}_class'] = merged.apply(class_func, axis=1)
        df_list.append(merged)
    stats_df = pd.concat(df_list).reset_index(drop=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.boxplot(
        y='dist_to_ref', x='country_code', data=stats_df, hue=f'{stat_name}_class', ax=ax, showfliers=False,
        palette=palette
    )
    ax.set_xlabel('Country')
    ax.set_ylabel(f'Distance to reference {feature_type}')
    plt.legend(loc='lower center', ncol=4, title=stat_name_long, bbox_to_anchor=(0.5, -0.3))
    plt.tight_layout()
    plt.savefig(
        f'D:/dem_comparison/figures/{stat_name}_vs_dist_to_ref_{feature_type}_by_class.png', dpi=300,
        bbox_inches='tight'
    )

# Plot histogram of statistic
for feature_type in feature_types:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for country_code, ax in zip(country_codes, axes.flatten()):
        dist_stats = pd.\
            concat(
            [pd.read_csv(utils.get_fp_to_stats(country_code, dem_name, feature_type, 'dist_to_ref')) for dem_name in
                    dem_names]).reset_index(drop=True)
        stats = pd.\
            concat(
            [pd.read_csv(utils.get_fp_to_stats(country_code, dem_name, feature_type, stat_name)) for dem_name in
                    dem_names]).reset_index(drop=True)
        merged = dist_stats.merge(stats[['point_id', stat_name]], how='left', on='point_id')
        merged = merged.sort_values(['dem_name', stat_name]).reset_index(drop=True)
        merged[stat_name] = merged[stat_name] * 100
        merged[stat_name].hist(ax=ax)
        ax.set_xlabel(stat_name_long)
        ax.set_ylabel(f'Number of {feature_type} point buffers')
        ax.set_title(country_code)
        plt.tight_layout()
        plt.savefig(
            f'D:/dem_comparison/figures/{stat_name}_hist_{feature_type}.png', dpi=300,
            bbox_inches='tight'
        )
