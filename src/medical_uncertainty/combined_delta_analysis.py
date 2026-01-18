#!/usr/bin/env python3
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

import starbars
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

pd_levels = [0.6, 0.8, 1.0, 'All']
color_map = {0.6: '#d62728', 0.8: '#ff7f0e', 1.0: '#2ca02c', 'All': 'gray'}
model_markers = {'Gemini 2.5 Pro': 'o', 'Gemini 3.0 Preview': 's', 'GPT 5.1': '^'}
pd_levels_symbol_maps = {
    "All":"="
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
#
# def create_combined_delta_f1_plot():
#     """Create plot showing all three models together"""
#     fig, ax = plt.subplots(figsize=(14, 10))
#
#
#     for model_name, json_path in models:
#         df = load_data(json_path)
#         marker = model_markers[model_name]
#
#         # Group by PA level and calculate averages
#         for pa_level in pd_levels:
#             pa_data = df[df['PA'] == pa_level]
#
#             if len(pa_data) == 0:
#                 continue
#
#             # Calculate delta (model - human) for each pathology
#             deltas = pa_data['agg_human - gemini-pro_mean_f1-score']
#             avg_delta = deltas.mean()
#             delta_min, delta_max = deltas.min(), deltas.max()
#             m_values = pa_data['m'].astype(float)
#             avg_m = m_values.mean()
#             m_min, m_max = m_values.min(), m_values.max()
#             total_samples = int(pa_data['S'].mean())
#
#             # Get color for this PA level
#             color = color_map[pa_level]
#
#             # Plot with model-specific marker and larger size scaling
#             circle_size= int(total_samples)*3
#             ax.scatter(avg_m, avg_delta, c=color, s=100, alpha=0.7,
#                       marker=marker, edgecolors=color, linewidth=1)
#
#             # Add sample count label
#             ax.text(avg_m  + 0.005, avg_delta, f'n={total_samples}',
#                    fontsize=8, ha='left', va='center')
#
#     # Add horizontal line at y=0 (no difference)
#     ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No difference')
#
#     # Create combined legend in upper left
#     pa_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color,
#                                 label=f'$p_d{pd_levels_symbol_maps.get(pa_level, '\\rightarrow')}${pa_level}')
#                   for pa_level, color in color_map.items()]
#
#     model_handles = [plt.scatter([], [], c='gray', s=100, alpha=0.7, marker=marker,
#                                 edgecolors='black', label=model_name)
#                     for model_name, marker in model_markers.items()]
#
#     # Combine all handles into one legend
#     all_handles = pa_handles + model_handles
#     ax.legend(handles=all_handles, loc='upper right', fontsize=10, ncol=2)
#
#     ax.set_xlabel('Average Positive class ratio (m)', fontsize=12, fontweight='bold')
#     ax.set_ylabel('Average Δ F1 (Human - Model)', fontsize=12, fontweight='bold')
#     # ax.set_title('Model Comparison: Average Performance Delta by PD Level\n(Circle size = total sample count)',
#     #             fontsize=14, fontweight='bold')
#     ax.grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     return fig

def create_combined_delta_accuracy_plot():
    """Create plot showing all three models together for accuracy"""
    fig, ax = plt.subplots(figsize=(14, 10))


    
    # Small offsets to separate overlapping points
    offsets = [-0.005, 0, 0.005]
    
    for i, (model_name, json_path) in enumerate(models):
        df = load_data(json_path)
        marker = model_markers[model_name]
        offset = offsets[i]
        
        # Group by PA level and calculate averages
        for pa_level in pd_levels:
            pa_data = df[df['PA'] == pa_level]
            
            if len(pa_data) == 0:
                continue

            #ax.set_xlim(0,0.7)
                
            # Calculate delta (model - human) for accuracy
            deltas = pa_data['agg_human - model_mean_accuracy']
            avg_delta = deltas.mean()
            m_values = pa_data['m'].astype(float)
            avg_m = m_values.mean()
            total_samples = int(pa_data['S'].mean())
            
            # Get color for this PA level
            color = color_map[pa_level]
            #
            # # Plot with model-specific marker
            # bplot = ax.boxplot(deltas,positions= [round(avg_m,2)])
            #
            #
            # for patch in bplot['boxes']:
            #     patch.set_gapcolor(color)
                # for pc in violin_parts['bodies']:
                #     pc.set_facecolor(color)
            ax.scatter(avg_m, avg_delta, c=color, s=total_samples/3, alpha=0.7,
                      marker=marker, edgecolors=color, linewidth=1)

            # ax.axvline(avg_m, ymin=min(deltas), ymax=max(deltas), c=color, alpha=0.7,
            #            marker=marker,  linewidth=1)
            
            # Add sample count label
            ax.text(avg_m + 0.005, avg_delta, f'n={total_samples}',
                   fontsize=8, ha='left', va='center')
    
    # Add horizontal line at y=0 (no difference)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No difference')
    
    # Create combined legend
    pa_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color,
                                label=f'$p_d{pd_levels_symbol_maps.get(pa_level, '\\rightarrow')}${pa_level}')
                  for pa_level, color in color_map.items()]
    
    model_handles = [plt.scatter([], [], c='gray', s=100, alpha=0.7, marker=marker,
                                edgecolors='black', label=model_name)
                    for model_name, marker in model_markers.items()]
    
    # Combine all handles into one legend
    all_handles = pa_handles + model_handles
    ax.legend(handles=all_handles, loc='upper right', fontsize=10, ncol=2)
    
    ax.set_xlabel('Average Positive class ratio (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Δ Accuracy (Human - Model)', fontsize=12, fontweight='bold')
    # ax.set_title('Model Comparison: Average Accuracy Delta by PD Level\n(Circle size = total sample count)',
    #             fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_combined_delta_f1_plot():
    """Create plot showing all three models together"""
    fig, ax = plt.subplots(figsize=(14, 10))



    for model_name, json_path in models:
        df = load_data(json_path)
        marker = model_markers[model_name]

        # Group by PA level and calculate averages
        for pa_level in pd_levels:
            pa_data = df[df['PA'] == pa_level]

            if len(pa_data) == 0:
                continue

            # Calculate delta (model - human) for each pathology
            deltas = pa_data['agg_human - model_mean_f1-score']
            avg_delta = deltas.mean()
            delta_min, delta_max = deltas.min(), deltas.max()
            m_values = pa_data['m'].astype(float)
            avg_m = m_values.mean()
            m_min, m_max = m_values.min(), m_values.max()
            total_samples = int(pa_data['S'].mean())

            # Get color for this PA level
            color = color_map[pa_level]

            # Plot with model-specific marker and larger size scaling
            circle_size = int(total_samples) * 3
            ax.scatter(avg_m, avg_delta, c=color, s=100, alpha=0.7,
                       marker=marker, edgecolors=color, linewidth=1)

            # Add sample count label
            ax.text(avg_m + 0.005, avg_delta, f'n={total_samples}',
                    fontsize=8, ha='left', va='center')

    # Add horizontal line at y=0 (no difference)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No difference')

    # Create combined legend in upper left
    pa_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color,
                                label=f'$p_d{pd_levels_symbol_maps.get(pa_level, '\\rightarrow')}${pa_level}')
                  for pa_level, color in color_map.items()]

    model_handles = [plt.scatter([], [], c='gray', s=100, alpha=0.7, marker=marker,
                                 edgecolors='black', label=model_name)
                     for model_name, marker in model_markers.items()]

    # Combine all handles into one legend
    all_handles = pa_handles + model_handles
    ax.legend(handles=all_handles, loc='upper right', fontsize=10, ncol=2)

    ax.set_xlabel('Average Positive class ratio (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Δ F1 (Human - Model)', fontsize=12, fontweight='bold')
    # ax.set_title('Model Comparison: Average Performance Delta by PD Level\n(Circle size = total sample count)',
    #             fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def boxplot_2d(x,y, ax, whis=1.5, color='black'):
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]

    ##the box
    box = Rectangle(
        (xlimits[0],ylimits[0]),
        (xlimits[2]-xlimits[0]),
        (ylimits[2]-ylimits[0]),
        ec = color,
        fc = color,
        alpha = 0.2,
        zorder=0
    )
    ax.add_patch(box)

    ##the x median
    vline = Line2D(
        [xlimits[1],xlimits[1]],[ylimits[0],ylimits[2]],
        color=color,
        zorder=1
    )
    ax.add_line(vline)

    ##the y median
    hline = Line2D(
        [xlimits[0],xlimits[2]],[ylimits[1],ylimits[1]],
        color=color,
        zorder=1
    )
    ax.add_line(hline)

    ##the central point
    ax.plot([xlimits[1]],[ylimits[1]], color=color, marker='o')

    ##the x-whisker
    ##defined as in matplotlib boxplot:
    ##As a float, determines the reach of the whiskers to the beyond the
    ##first and third quartiles. In other words, where IQR is the
    ##interquartile range (Q3-Q1), the upper whisker will extend to
    ##last datum less than Q3 + whis*IQR). Similarly, the lower whisker
    ####will extend to the first datum greater than Q1 - whis*IQR. Beyond
    ##the whiskers, data are considered outliers and are plotted as
    ##individual points. Set this to an unreasonably high value to force
    ##the whiskers to show the min and max values. Alternatively, set this
    ##to an ascending sequence of percentile (e.g., [5, 95]) to set the
    ##whiskers at specific percentiles of the data. Finally, whis can
    ##be the string 'range' to force the whiskers to the min and max of
    ##the data.
    iqr = xlimits[2]-xlimits[0]

    ##left
    left = np.min(x[x > xlimits[0]-whis*iqr])
    whisker_line = Line2D(
        [left, xlimits[0]], [ylimits[1],ylimits[1]],
        color = color,
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [left, left], [ylimits[0],ylimits[2]],
        color = color,
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##right
    right = np.max(x[x < xlimits[2]+whis*iqr])
    whisker_line = Line2D(
        [right, xlimits[2]], [ylimits[1],ylimits[1]],
        color = color,
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [right, right], [ylimits[0],ylimits[2]],
        color = color,
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##the y-whisker
    iqr = ylimits[2]-ylimits[0]

    ##bottom
    bottom = np.min(y[y > ylimits[0]-whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [bottom, ylimits[0]],
        color = color,
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [bottom, bottom],
        color = color,
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##top
    top = np.max(y[y < ylimits[2]+whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [top, ylimits[2]],
        color = color,
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [top, top],
        color = color,
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##outliers
    mask = (x<left)|(x>right)|(y<bottom)|(y>top)
    ax.scatter(
        x[mask],y[mask],
        facecolors=color, edgecolors=color
    )


def create_combined_delta_doublebox_accuracy_plot():
    """Create plot showing all three models together for accuracy"""
    fig, ax = plt.subplots(figsize=(20, 10))





    dfs = []
    for i, (model_name, json_path) in enumerate(models):
        df_m = load_data(json_path)
        df_m["model"] = model_name
        df_m["model_marker"] = model_markers[model_name]
        dfs.append(df_m)

    df = pd.concat(dfs)




    # Group by PA level and calculate averages
    for pa_level in pd_levels:
        if pa_level == "All": continue
        pa_data = df[df['PA'] == pa_level]

        if len(pa_data) == 0:
            continue

        # ax.set_xlim(0,0.7)

        # Calculate delta (model - human) for accuracy
        deltas = pa_data['agg_human - model_mean_accuracy']
        m_values = pa_data['m'].astype(float)
        ax.set_xlim(0.0, 0.8)
        ax.set_ylim(-0.3, 0.6)

        # Get color for this PA level
        color = color_map[pa_level]
        #
        # # Plot with model-specific marker
        # bplot = ax.boxplot(deltas,positions= [round(avg_m,2)])
        #
        #
        # for patch in bplot['boxes']:
        #     patch.set_gapcolor(color)
        # for pc in violin_parts['bodies']:
        #     pc.set_facecolor(color)
        boxplot_2d(m_values,deltas,ax, color=color )

    # Add horizontal line at y=0 (no difference)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No difference')

    # Create combined legend
    pa_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color,
                                label=f'$p_d{pd_levels_symbol_maps.get(pa_level, '\\rightarrow')}${pa_level}')
                  for pa_level, color in color_map.items()]



    # Combine all handles into one legend
    all_handles = pa_handles
    ax.legend(handles=all_handles, loc='upper right', fontsize=10, ncol=2)

    ax.set_xlabel('Positive class ratio (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Δ Accuracy (Human - Model)', fontsize=12, fontweight='bold')
    # ax.set_title('Model Comparison: Average Accuracy Delta by PD Level\n(Circle size = total sample count)',
    #             fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def create_combined_delta_doublebox_f1_plot():
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




    # Group by PA level and calculate averages
    for pa_level in pd_levels:
        if pa_level == "All": continue
        pa_data = df[df['PA'] == pa_level]

        if len(pa_data) == 0:
            continue

        # ax.set_xlim(0,0.7)

        # Calculate delta (model - human) for f1
        deltas = pa_data['agg_human - model_mean_f1-score']
        m_values = pa_data['m'].astype(float)
        ax.set_xlim(0.0, 0.8)
        ax.set_ylim(-0.3, 0.6)

        # Get color for this PA level
        color = color_map[pa_level]
        #
        # # Plot with model-specific marker
        # bplot = ax.boxplot(deltas,positions= [round(avg_m,2)])
        #
        #
        # for patch in bplot['boxes']:
        #     patch.set_gapcolor(color)
        # for pc in violin_parts['bodies']:
        #     pc.set_facecolor(color)
        boxplot_2d(m_values,deltas,ax, color=color )

    # Add horizontal line at y=0 (no difference)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No difference')

    # Create combined legend
    pa_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color,
                                label=f'$p_d{pd_levels_symbol_maps.get(pa_level, '\\rightarrow')}${pa_level}')
                  for pa_level, color in color_map.items()]



    # Combine all handles into one legend
    all_handles = pa_handles
    ax.legend(handles=all_handles, loc='upper right', fontsize=10, ncol=2)

    ax.set_xlabel('Positive class ratio (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Δ F1 (Human - Model)', fontsize=12, fontweight='bold')
    # ax.set_title('Model Comparison: Average Accuracy Delta by PD Level\n(Circle size = total sample count)',
    #             fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def create_combined_simple_boxplot_seaborn(column_name, metric_label):


    sns.set_style("darkgrid")
    sns.set(font_scale=2.5)
    sns.set(rc={'axes.facecolor': 'none', 'figure.facecolor': 'none', "grid.color":"lightgray", "axes.edgecolor":"black"})

    fig, ax = plt.subplots(figsize=(20, 10))






    dfs = []
    for i, (model_name, json_path) in enumerate(models):
        df_m = load_data(json_path)
        df_m["model"] = model_name
        df_m["model_marker"] = model_markers[model_name]
        dfs.append(df_m)

    df = pd.concat(dfs)

    pvalues = []

    # Group by PA level and calculate averages
    all_data_labels = []
    # Define median properties with a specific color and linewidth
    median_properties = dict(color='black', linewidth=2.5)
    df_fmt = pd.DataFrame()
    for pa_level in pd_levels:
        # if pa_level == "All": continue
        print(pa_level)
        pa_data = df[df['PA'] == pa_level]
        deltas = pa_data[column_name].values

        df_fmt[str(pa_level)] = deltas
        all_data_labels.append(str(pa_level))

    for p in itertools.combinations(all_data_labels, 2):
        if p[0]=="All" or p[1]=="All": continue


        ttest_result = ttest_ind(df_fmt[p[0]], df_fmt[p[1]], nan_policy="omit", equal_var=True)
        pvalues.append(
            (p[0], p[1], ttest_result.pvalue)
        )

    x_label = "$p_d$"
    y_label= f"$\\Delta$ = $H_{{{metric_label}}} - M_{{{metric_label}}}$"
    sns.boxplot(data=df_fmt,  ax=ax, fill = True, showmeans=True, order=all_data_labels, palette={str(k):v for k,v in color_map.items()}, meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"white"})

    # adding statistical annotation
    ax.set_ylabel(y_label, fontsize=18, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18)
    annotations = pvalues
    starbars.draw_annotation(annotations, ax=ax, fontsize=18)
    ax.set_ylim(-0.3, 1.0)

    return fig

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
        #if pa_level == "All": continue
        print(pa_level)
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
                    "effect_size": abs(df_pd1.mean() - df_pdnot1.mean() ),
                    "metric": metric_label,
                }
            )






        # if len(pa_data) == 0:
    #     continue

    # ax.set_xlim(0,0.7)

    # Calculate delta (model - human) for f1
    #ax.set_xlim(0.0, 0.8)
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
        #boxplot_2d(deltas,ax=ax, color=color )



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

#
# def create_combined_simple_boxplot_human_vs_models(column_name_human,column_name_model, metric_label):
#     """Create plot showing all three models together for accuracy"""
#     fig, ax = plt.subplots(figsize=(20, 10))
#
#     models = [
#         # ('Gemini 2.5 Pro',
#         #  Path(__file__).parent.parent.parent / "data_dir/checkpoints/gemini25pro/scores_exclude_small_samples.json"),
#
#         ('Gemini 3.0 Preview', Path(
#             __file__).parent.parent.parent / "data_dir/checkpoints/gemini30preview/scores_exclude_small_samples.json"),
#         # ('GPT 5.1',
#         #  Path(__file__).parent.parent.parent / "data_dir/checkpoints/gpt-5.1/scores_exclude_small_samples.json")
#
#     ]
#
#     # just use anyone as human performance is the same.
#     human_data = Path(__file__).parent.parent.parent / "data_dir/checkpoints/gemini25pro/scores_exclude_small_samples.json"
#
#
#
#     df_h = load_data(human_data)
#
#     # Small offsets to separate overlapping points
#     offsets = [-0.005, 0, 0.005]
#
#     dfs = []
#     for i, (model_name, json_path) in enumerate(models):
#         df_m = load_data(json_path)
#         df_m["model"] = model_name
#         df_m["model_marker"] = model_markers[model_name]
#         dfs.append(df_m)
#
#     df = pd.concat(dfs)
#
#     pvalues = []
#
#     # Group by PA level and calculate averages
#     all_data = []
#     all_data_props = []
#     # Define median properties with a specific color and linewidth
#     median_properties = dict(color='black', linewidth=2.5)
#     for pa_level in pd_levels:
#         #if pa_level == "All": continue
#         print(pa_level)
#         pa_data = df[df['PA'] == pa_level]
#         score_values = pa_data[column_name_model]
#
#         all_data.append(score_values)
#         all_data_props.append(
#             {"pa_level": pa_level,
#              "type": "M", # model
#              "hatch": None
#              }
#
#         )
#
#     for pa_level in pd_levels:
#
#         # add human
#         all_data.append(df_h[df_h['PA'] == pa_level][column_name_human])
#         all_data_props.append(
#             {"pa_level": pa_level,
#              "type": "H",  # model
#              "hatch": "."
#              }
#
#         )
#
#
#         df_pd = df[df["PA"] == pa_level][column_name_model]
#         dfh_pd = df_h[df_h["PA"] == pa_level][column_name_human]
#
#         ttest_result = ttest_ind(df_pd, dfh_pd, nan_policy="omit", equal_var=True)
#         pvalues.append(
#             {
#                 "basecompare": pa_level,
#                 "p-value": ttest_result.pvalue,
#                 "effect_size": abs(dfh_pd.mean() - df_pd.mean() ),
#                 "metric": metric_label,
#             }
#         )
#
#
#
#
#
#
#         # if len(pa_data) == 0:
#     #     continue
#
#     # ax.set_xlim(0,0.7)
#
#     # Calculate delta (model - human) for f1
#     #ax.set_xlim(0.0, 0.8)
#     ax.set_ylim(0.0, 1.01)
#     print(pvalues)
#
#
#     #
#     # # Plot with model-specific marker
#     bplot = ax.boxplot(all_data, medianprops=median_properties, showmeans=True, patch_artist=True)
#     for i in range(len(all_data)):
#         box = bplot['boxes'][i]
#         box_x = []
#         box_y = []
#
#         box.set(facecolor= color_map[all_data_props[i]["pa_level"]], alpha=0.7)
#         # change hatch
#         if all_data_props[i]["hatch"]:
#             box.set(hatch=all_data_props[i]["hatch"])
#
#         # for j in range(5):
#         #     box_x.append(box.get_xdata()[j])
#         #     box_y.append(box.get_ydata()[j])
#         # box_coords = np.column_stack([box_x, box_y])
#         # ax.add_patch(Polygon(box_coords, facecolor=color_map[all_data_props[i]["pa_level"]], alpha=0.7))
#         #
#
#         # for patch in bplot['boxes']:
#         #     patch.set_gapcolor(color)
#         # for pc in violin_parts['bodies']:
#         #     pc.set_facecolor(color)
#         #boxplot_2d(deltas,ax=ax, color=color )
#
#
#
#     # Add horizontal line at y=0 (no difference)
#     ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No difference')
#
#     # # Create combined legend
#     # pa_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color,
#     #                             label=f'$p_d{pd_levels_symbol_maps.get(pa_level, '\\rightarrow')}${pa_level}')
#     #               for pa_level, color in color_map.items()]
#
#
#
#     # Combine all handles into one legend
#     # all_handles = pa_handles
#     # ax.legend(handles=all_handles, loc='upper right', fontsize=10, ncol=2)
#     ax.set_xticklabels([f"{prop['type']} $p_d{pd_levels_symbol_maps.get(prop['pa_level'], '\\rightarrow')}${prop['pa_level']}"
#                    for prop in all_data_props])
#     ax.set_ylabel(f'Δ {metric_label}', fontsize=14, fontweight='bold')
#     # ax.set_title('Model Comparison: Average Accuracy Delta by PD Level\n(Circle size = total sample count)',
#     #             fontsize=14, fontweight='bold')
#     ax.grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     return fig

def create_combined_delta_simple_boxplot_f1():
    return create_combined_simple_boxplot_seaborn('agg_human - model_mean_f1-score', "F1")


def create_combined_delta_simple_boxplot_accuracy():
    return create_combined_simple_boxplot_seaborn('agg_human - model_mean_accuracy', "Accuracy")

#
# def create_combined_simple_boxplot_human_accuracy():
#     return create_combined_simple_boxplot_human_vs_models('agg_human_mean_accuracy', "gemini-pro_accuracy", "Accuracy")
#
#
# def create_combined_simple_boxplot_human_f1():
#     return create_combined_simple_boxplot_human_vs_models('agg_human_mean_f1-score',"gemini-pro_f1-score", "F1")
#


def main():
    print("Creating Combined Delta F1 and Accuracy Analysis...")
    
    output_dir = Path(__file__).parent.parent.parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # F1 plot
    fig = create_combined_delta_f1_plot()
    fig.savefig(output_dir / "combined_delta_f1_analysis.png", dpi=400, bbox_inches='tight')
    plt.close(fig)
    print("Combined Delta F1 Analysis visualization saved")
    
    # Accuracy plot
    fig = create_combined_delta_accuracy_plot()
    fig.savefig(output_dir / "combined_delta_accuracy_analysis.png", dpi=400, bbox_inches='tight')
    plt.close(fig)
    print("Combined Delta Accuracy Analysis visualization saved")

    fig =  create_combined_delta_doublebox_accuracy_plot()
    fig.savefig(output_dir / "combined_2dbox_accuracy_analysis.png", dpi=400, bbox_inches='tight')
    plt.close(fig)
    print("Combined 2d Delta Accuracy Analysis visualization saved")

    fig = create_combined_delta_doublebox_f1_plot()
    fig.savefig(output_dir / "combined_2dbox_f1_analysis.png", dpi=400, bbox_inches='tight')
    plt.close(fig)
    print("Combined 2d Delta F1 Analysis visualization saved")

    fig =     create_combined_delta_simple_boxplot_f1()
    fig.savefig(output_dir / "combined_simple_boxplot_delte_f1_analysis.png", dpi=400, bbox_inches='tight')
    plt.close(fig)
    print("Combined simple boxplot Delta F1 Analysis visualization saved")

    fig = create_combined_delta_simple_boxplot_accuracy()
    fig.savefig(output_dir / "combined_simple_boxplot_delte_accuracy_analysis.png", dpi=400, bbox_inches='tight')
    plt.close(fig)
    print("Combined simple boxplot Delta accuracy Analysis visualization saved")

    #
    # fig = create_combined_simple_boxplot_human_accuracy()
    # fig.savefig(output_dir / "combined_simple_boxplot_human_accuracy_analysis.png", dpi=400, bbox_inches='tight')
    # plt.close(fig)
    # print("Combined simple boxplot human vs model accuracy Analysis visualization saved")
    #
    # fig = create_combined_simple_boxplot_human_f1()
    # fig.savefig(output_dir / "combined_simple_boxplot_human_f1_analysis.png", dpi=400, bbox_inches='tight')
    # plt.close(fig)
    # print("Combined simple boxplot human vs model F1 Analysis visualization saved")


if __name__ == "__main__":
    main()
