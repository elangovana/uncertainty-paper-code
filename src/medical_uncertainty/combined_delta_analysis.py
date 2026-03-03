#!/usr/bin/env python3
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

import starbars
from scipy.stats import ttest_ind, ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from scipy import stats
import statistics

pd_levels = [0.6, 0.8, 1.0, 'All']
color_map = {0.6: '#d62728', 0.8: '#ff7f0e', 1.0: '#2ca02c', 'All': 'gray'}
model_markers = {'Gemini 2.5 Pro': 'o', 'Gemini 3.0 Preview': 's', 'GPT 5.1': '^'}
pd_levels_symbol_maps = {
    "All": "="
}

models = [
    ('Gemini 2.5 Pro', Path(
        __file__).parent.parent.parent / "data_dir/checkpoints/gemini25pro/gemini25pro_scores_exclude_small_samples.json"),
    ('Gemini 3.0 Preview', Path(
        __file__).parent.parent.parent / "data_dir/checkpoints/gemini30preview/gemini30preview_scores_exclude_small_samples.json"),
    ('GPT 5.1',
     Path(__file__).parent.parent.parent / "data_dir/checkpoints/gpt-5.1/gpt-5.1_scores_exclude_small_samples.json")
]


def load_data(json_path):
    """Load data from JSON checkpoint"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def create_combined_simple_box_plot_by_df(df, column_name, metric_label, y_label=None):
    sns.set_style("darkgrid")
    sns.set(font_scale=2.5)
    sns.set(
        rc={'axes.facecolor': 'none', 'figure.facecolor': 'none', "grid.color": "lightgray", "axes.edgecolor": "black"}
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    mean_p_values = []
    variance_p_values = []

    # Group by PA level and calculate averages
    all_data_labels = []
    # Define median properties with a specific color and linewidth
    median_properties = dict(color='black', linewidth=2.5)
    df_fmt = pd.DataFrame()
    for pa_level in pd_levels:
        # if pa_level == "All": continue
        # print(pa_level, df['PA'])
        pa_data = df[df['PA'] == pa_level]
        deltas = pa_data[column_name].values

        df_fmt[str(pa_level)] = deltas
        all_data_labels.append(str(pa_level))

    for p in itertools.combinations(all_data_labels, 2):
        if p[0] == "All" or p[1] == "All": continue

        ttest_result = ttest_ind(df_fmt[p[0]], df_fmt[p[1]], nan_policy="omit", equal_var=False)
        mean_p_values.append(
            (p[0], p[1], ttest_result.pvalue)
        )
        stat, variance_p_value = stats.levene(df_fmt[p[0]], df_fmt[p[1]], nan_policy="omit", center="trimmed")
        variance_p_values.append((p[0], p[1], variance_p_value))

    x_label = "$p_d$"
    if not y_label:
        y_label = f"$\\Delta$ = $H_{{{metric_label}}} - M_{{{metric_label}}}$"
    sns.boxplot(data=df_fmt, ax=ax, fill=True, showmeans=True, order=all_data_labels,
                palette={str(k): v for k, v in color_map.items()},
                meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "white"})

    # adding statistical annotation
    ax.set_ylabel(y_label, fontsize=20, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=20)
    starbars.draw_annotation(mean_p_values, ax=ax, fontsize=20)
    starbars.draw_annotation(variance_p_values, ax=ax, fontsize=20, color='red')

    ax.set_ylim(-0.3, 1.3)
    return fig


def create_combined_paired_delta_box_plot_by_df(df, column_name_1, column_name_2, metric_label, y_label=None):
    sns.set_style("darkgrid")
    sns.set(font_scale=2.5)
    sns.set(
        rc={'axes.facecolor': 'none', 'figure.facecolor': 'none', "grid.color": "lightgray", "axes.edgecolor": "black"}
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    mean_p_values = {}
    max_delta_by_pa = {}


    # Group by PA level and calculate averages
    all_data_labels = []

    df_fmt = pd.DataFrame()
    for pa_level in pd_levels:
        pa_data = df[df['PA'] == pa_level]
        deltas = list(pa_data[column_name_1] - pa_data[column_name_2])


        df_fmt[str(pa_level)] = deltas
        all_data_labels.append(str(pa_level))

        ttest_result = ttest_rel(pa_data[column_name_1], pa_data[column_name_2], nan_policy="omit")
        mean_p_values[str(pa_level)] = ttest_result.pvalue
        max_delta_by_pa[str(pa_level)] = max(deltas)

    x_label = "$p_d$"
    if not y_label:
        y_label = f"$\\Delta$ = $H_{{{metric_label}}} - M_{{{metric_label}}}$"
    boxplt = sns.boxplot(data=df_fmt, ax=ax, fill=True, showmeans=True, order=all_data_labels,
                         palette={str(k): v for k, v in color_map.items()},
                         meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "white"})

    for i, pa in enumerate(all_data_labels):
        paired_p_val = mean_p_values[pa]
        if paired_p_val > 0.05:
            paired_p_val = f"ns ({paired_p_val:0.2f})"
        elif 0.01 <paired_p_val <= 0.5:
            paired_p_val =f"*"
        elif 0.001 <paired_p_val <= 0.01:
            paired_p_val =f"**"
        elif 0.0001 <paired_p_val <= 0.001:
            paired_p_val =f"***"
        else:
            paired_p_val = f"****"
        #boxplt.annotate(paired_p_val, xy=(i, max_delta_by_pa[pa]+0.1), horizontalalignment='center', annot_kws={"fontsize": 20})
        boxplt.text(i, max_delta_by_pa[pa] + 0.1, paired_p_val,horizontalalignment='center',
                    size='large')

    # adding statistical annotation
    ax.set_ylabel(y_label, fontsize=20, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=20)

    ax.set_ylim(-0.3, 0.6)
    return fig


def create_combined_simple_boxplot_seaborn(column_name, metric_label):
    dfs = []
    for i, (model_name, json_path) in enumerate(models):
        df_m = load_data(json_path)
        df_m["model"] = model_name
        df_m["model_marker"] = model_markers[model_name]
        dfs.append(df_m)

    df = pd.concat(dfs)

    return create_combined_simple_box_plot_by_df(df, column_name, metric_label)


def create_combined_simple_boxplot(column_name, metric_label):
    """Create plot showing all three models together for accuracy"""
    fig, ax = plt.subplots(figsize=(20, 10))

    # Small offsets to separate overlapping points
    offsets = [-0.005, 0, 0.005]

    dfs = []
    for i, (model_name, json_path) in enumerate(models):
        df_m = load_data(json_path)
        df_m["model"] = model_name
        df_m["model_marker"] = model_markers[model_name]
        dfs.append(df_m)

    df = pd.concat(dfs)

    pvalues = []

    # Group by PA level and calculate averages
    all_data = []
    all_data_labels = []
    # Define median properties with a specific color and linewidth
    median_properties = dict(color='black', linewidth=2.5)
    for pa_level in pd_levels:
        # if pa_level == "All": continue
        pa_data = df[df['PA'] == pa_level]
        deltas = pa_data[column_name]

        all_data.append(deltas)
        all_data_labels.append(pa_level)

        if pa_level not in ["All", 1.0]:
            df_pd1 = df[df["PA"] == 1.0][column_name]
            df_pdnot1 = df[df["PA"] == pa_level][column_name]
            ttest_result = ttest_ind(df_pd1, df_pdnot1, nan_policy="omit", equal_var=True)
            pvalues.append(
                {
                    "basecompare": pa_level,
                    "p-value": ttest_result.pvalue,
                    "effect_size": abs(df_pd1.mean() - df_pdnot1.mean()),
                    "metric": metric_label,
                }
            )

        # if len(pa_data) == 0:
    #     continue

    # ax.set_xlim(0,0.7)

    # Calculate delta (model - human) for f1
    # ax.set_xlim(0.0, 0.8)
    ax.set_ylim(-0.3, 0.6)

    #
    # # Plot with model-specific marker
    bplot = ax.boxplot(all_data, medianprops=median_properties, showmeans=True)
    for i in range(len(all_data)):
        box = bplot['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        ax.add_patch(Polygon(box_coords, facecolor=color_map[all_data_labels[i]], alpha=0.7))

        # for patch in bplot['boxes']:
        #     patch.set_gapcolor(color)
        # for pc in violin_parts['bodies']:
        #     pc.set_facecolor(color)
        # boxplot_2d(deltas,ax=ax, color=color )

    # Add horizontal line at y=0 (no difference)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No difference')

    # # Create combined legend
    # pa_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color,
    #                             label=f'$p_d{pd_levels_symbol_maps.get(pa_level, '\\rightarrow')}${pa_level}')
    #               for pa_level, color in color_map.items()]

    # Combine all handles into one legend
    # all_handles = pa_handles
    # ax.legend(handles=all_handles, loc='upper right', fontsize=10, ncol=2)
    ax.set_xticklabels([f'$p_d{pd_levels_symbol_maps.get(pa_level, '\\rightarrow')}${pa_level}'
                        for pa_level in color_map])
    ax.set_ylabel(f'Δ {metric_label} (Human - Model)', fontsize=14, fontweight='bold')
    # ax.set_title('Model Comparison: Average Accuracy Delta by PD Level\n(Circle size = total sample count)',
    #             fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_combined_delta_simple_boxplot_f1():
    return create_combined_simple_boxplot_seaborn('agg_human - model_mean_f1-score', "F1")


def create_combined_delta_simple_boxplot_accuracy():
    return create_combined_simple_boxplot_seaborn('agg_human - model_mean_accuracy', "Accuracy")


def main():
    print("Creating Combined Delta F1 and Accuracy Analysis...")

    output_dir = Path(__file__).parent.parent.parent / "visualizations"
    output_dir.mkdir(exist_ok=True)

    fig = create_combined_delta_simple_boxplot_f1()
    fig.savefig(output_dir / "combined_simple_boxplot_delte_f1_analysis.png", dpi=400, bbox_inches='tight')
    plt.close(fig)
    print("Combined simple boxplot Delta F1 Analysis visualization saved")

    fig = create_combined_delta_simple_boxplot_accuracy()
    fig.savefig(output_dir / "combined_simple_boxplot_delte_accuracy_analysis.png", dpi=400, bbox_inches='tight')
    plt.close(fig)
    print("Combined simple boxplot Delta accuracy Analysis visualization saved")


if __name__ == "__main__":
    main()
