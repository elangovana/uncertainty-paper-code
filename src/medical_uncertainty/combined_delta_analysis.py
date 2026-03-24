#!/usr/bin/env python3
import itertools

import pandas as pd
import json
from pathlib import Path

import starbars
from scipy.stats import ttest_ind, ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns


from scipy import stats

pd_levels = [0.6, 0.8, 1.0, 'All']
color_map = {0.6: '#d62728', 0.8: '#ff7f0e', 1.0: '#2ca02c', 'All': 'gray'}
pd_levels_symbol_maps = {
    "All": "="
}

models = [
    ('Gemini 2.5 Pro', Path(
        __file__).parent.parent.parent / "data_dir/checkpoints/gemini25pro/gemini25pro_scores_exclude_small_samples.json"),
    ('Gemini 3.0 Preview', Path(
        __file__).parent.parent.parent / "data_dir/checkpoints/gemini30preview/gemini30preview_scores_exclude_small_samples.json"),
    ('Gemini 3.1 Pro', Path(
        __file__).parent.parent.parent / "data_dir/checkpoints/gemini31propreview/gemini31propreview_scores_exclude_small_samples.json"),
    ('GPT 5.1',
     Path(__file__).parent.parent.parent / "data_dir/checkpoints/gpt-5.1/gpt-5.1_scores_exclude_small_samples.json")
]


def load_data(json_path):
    """Load data from JSON checkpoint"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def create_mean_variance_pd_comparer_boxplot_by_df(df, column_name, metric_label, y_label=None):
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

    df_fmt = pd.DataFrame()
    for pa_level in pd_levels:

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
        stat, variance_p_value = stats.levene(df_fmt[p[0]], df_fmt[p[1]], nan_policy="omit", center="trimmed", proportiontocut=0.10)
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
    starbars.draw_annotation(mean_p_values, ax=ax, fontsize=20, bar_gap=0.01)
    starbars.draw_annotation(variance_p_values, ax=ax, fontsize=20, bar_gap=0.01, color='red')

    ax.set_ylim(-0.3, 1.3)
    return fig


def create_paired_comparer_boxplot_by_df(df, column_name_1, column_name_2, metric_label, y_label=None):
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
                    fontsize=20)

    # adding statistical annotation
    ax.set_ylabel(y_label, fontsize=20, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=20)

    ax.set_ylim(-0.3, 0.6)
    return fig


def create_multimodel_mean_variance_pd_comparer_boxplot_by_df(column_name, metric_label):
    dfs = []
    for i, (model_name, json_path) in enumerate(models):
        df_m = load_data(json_path)
        df_m["model"] = model_name
        #df_m["model_marker"] =  [model_name]
        dfs.append(df_m)

    df = pd.concat(dfs)

    return create_mean_variance_pd_comparer_boxplot_by_df(df, column_name, metric_label)



def create_multimodel_mean_variance_pd_delta_h_vs_allmodels_f1_comparer_boxplot():
    return create_multimodel_mean_variance_pd_comparer_boxplot_by_df('agg_human - model_mean_f1-score', "F1")


def create_multimodel_mean_variance_pd_delta_h_vs_allmodels_acc_comparer_boxplot():
    return create_multimodel_mean_variance_pd_comparer_boxplot_by_df('agg_human - model_mean_accuracy', "Accuracy")


def main():
    print("Creating Combined Delta F1 and Accuracy Analysis...")

    output_dir = Path(__file__).parent.parent.parent / "visualizations"
    output_dir.mkdir(exist_ok=True)

    fig = create_multimodel_mean_variance_pd_delta_h_vs_allmodels_f1_comparer_boxplot()
    fig.savefig(output_dir / "combined_simple_boxplot_delte_f1_analysis.png", dpi=400, bbox_inches='tight')
    plt.close(fig)
    print("Combined simple boxplot Delta F1 Analysis visualization saved")

    fig = create_multimodel_mean_variance_pd_delta_h_vs_allmodels_acc_comparer_boxplot()
    fig.savefig(output_dir / "combined_simple_boxplot_delte_accuracy_analysis.png", dpi=400, bbox_inches='tight')
    plt.close(fig)
    print("Combined simple boxplot Delta accuracy Analysis visualization saved")


if __name__ == "__main__":
    main()
