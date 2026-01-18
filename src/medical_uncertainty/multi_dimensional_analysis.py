#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

pd_levels = [0.6, 0.8, 1.0, 'All']
pd_colors = ['#d62728', '#ff7f0e', '#2ca02c', 'gray']
pd_levels_symbol_maps = {
    "All":"="
}

# Create explicit mapping for PA values to colors
color_map = {0.6: '#d62728', 0.8: '#ff7f0e', 1.0: '#2ca02c', 'All': 'gray'}

# Cre

# Create proper abbreviations
abbreviations = {
    'Atelectasis': 'At',
    'Cardiomegaly': 'Ca',
    'Consolidation': 'Co',
    'Edema': 'Ed',
    'Enlarged Cardiomediastinum': 'EC',
    'Lung Opacity': 'LO',
    'Pleural Effusion': 'PE',
    'Pneumonia': 'Pn',
    'Support Devices': 'SD'
}
pa_pd_map = {
            1.0:0.99
        }

random_labeller_pd = 0.5
pathology_handles = [plt.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='none',
                                   label=f'{name}($\\bf{{{sb}}}$)')
                     for name, sb in abbreviations.items()]


def load_data(json_path):
    """Load Gemini Pro 2.5 data from JSON checkpoint"""

    with open(json_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # # Filter for the 8 pathologies we need (excluding Fracture and No Finding)
    # pathologies = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    #                'Enlarged Cardiomediastinum', 'Lung Opacity', 'Pleural Effusion',
    #                'Pneumonia', 'Support Devices']

    # df_filtered = df[df['name'].isin(pathologies)].copy()

    return df

def cal_f1_given_p_m( p, m):
    y =np.divide (2 * m * p,(2 * m * p + 1-p))
    return y
# # ---> Gemini Pro 3.0  
def create_f1_vs_class_imbalance_plot(json_path, model_name):
    """Create multi-dimensional analysis plot for Gemini Pro 2.5 data"""
    df = load_data(json_path)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create plot with all data points
    for _, row in df.iterrows():
        pathology = row['name']
        pa_level = row['PA']
        f1_val = row['model_f1-score']
        m_val = float(row['m'])
        human_f1 = row['agg_human_mean_f1-score']

        # Get color for this PA level
        color = color_map[pa_level]

        # Add Gemini dot with PA level color and x shape
        ax.scatter(m_val, f1_val, c=color, s=40, alpha=0.8, marker='x')

        # Add human F1 dot in black with circle shape
        ax.scatter(m_val, human_f1, c=color, s=50, alpha=0.8, marker='o')

        # Add arrow from Human to Gemini
        ax.annotate('', xy=(m_val, f1_val), xytext=(m_val, human_f1),
                    arrowprops=dict(arrowstyle='->', color=color, alpha=0.8, lw=2))

        # 2. Define the formula to calculate the y-values

        if pa_level != "All":
            x = np.linspace(0.001, 0.75, 100)

            ax.plot(x, cal_f1_given_p_m(pa_pd_map.get(pa_level, pa_level), x),
            color = color, linestyle = '--', linewidth = 1, alpha = 0.7)


        # Add vertical text at the higher F1 position
        higher_f1 = max(f1_val, human_f1)
        ax.text(m_val - 0.002, higher_f1 + 0.02, abbreviations[pathology], ha='left', va='bottom',
                fontsize=6, fontweight='bold', color=color, rotation=0)

    x = np.linspace(0.001, 0.75, 100)

    ax.plot(x, cal_f1_given_p_m(random_labeller_pd, x),
            color="steelblue", linestyle='dashdot',  linewidth=3, alpha=0.7)

    random_baseline_handle, = plt.plot([], [], c='steelblue', linestyle='dashdot', linewidth=2,
                                       label='Random Baseline F1')

    # Add center lines at m=0.2 and F1=0.8
    # ax.axvline(x=0.2, color='black', linestyle='', linewidth=1, alpha=0.7)


    # Add quadrant labels
    # ax.text(0.15, 1, 'Low m\nHigh F1', ha='center', va='center', fontsize=12,
    #         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    # ax.text(0.25, 1, 'High m\nHigh F1', ha='center', va='center', fontsize=12,
    #         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    # ax.text(0.15, 0.2, 'Low m\nLow F1', ha='center', va='center', fontsize=12,
    #         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    # ax.text(0.25, 0.2, 'High m\nLow F1', ha='center', va='center', fontsize=12,
    #         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))

    # Create legend for PA levels

    pa_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color,
                                label=f'$p_d{pd_levels_symbol_maps.get(pa_level, '\\rightarrow')}${pa_level}')
                  for pa_level, color in color_map.items()]

    baseline_handle, = plt.plot([], [], c='gray', linestyle='--',
                                 label='Expected F1')



    # Add human F1 to legend
    human_handles = [plt.scatter([], [], c='gray', s=30, marker='o',
                                 label='Human F1')]

    model_handles = [plt.scatter([], [], c='gray', s=30, marker='x',
                                 label=model_name)]

    all_handles = pa_handles + human_handles + model_handles + [baseline_handle, random_baseline_handle] + pathology_handles
    ax.legend(handles=all_handles, loc='lower right', ncol=2, fontsize=8)

    ax.set_xlabel('Positive class ratio (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name}: Multidimensional Analysis with Expected F1', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 0.72)
    ax.set_ylim(-0.01, 1.1)

    plt.tight_layout()
    return fig


def create_accuracy_vs_class_imbalance_plot(json_path, model_name):
    """Create multi-dimensional analysis plot for Gemini Pro 2.5 accuracy data"""
    df = load_data(json_path)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create plot with all data points
    for _, row in df.iterrows():
        pathology = row['name']
        pa_level = row['PA']
        acc_val = row['model_accuracy']
        m_val = float(row['m'])
        human_acc = row['agg_human_mean_accuracy']

        # Get color for this PA level
        color = color_map[pa_level]

        # Add Gemini dot with PA level color and x shape
        ax.scatter(m_val, acc_val, c=color, s=40, alpha=0.8, marker='x')

        # Add human accuracy dot in black with circle shape
        ax.scatter(m_val, human_acc, c=color, s=50, alpha=0.8, marker='o')

        # Add arrow from Human to Gemini
        ax.annotate('', xy=(m_val, acc_val), xytext=(m_val, human_acc),
                    arrowprops=dict(arrowstyle='->', color=color, alpha=0.8, lw=2))

        if pa_level != "All":
            x = np.linspace(0.001, 0.75, 100)

            ax.plot(x, np.full(x.shape, pa_pd_map.get(pa_level, pa_level)),
            color = color, linestyle = '--', linewidth = 1, alpha = 0.7)

        # Add vertical text at the higher accuracy position
        higher_acc = max(acc_val, human_acc)
        ax.text(m_val - 0.002, higher_acc + 0.02, abbreviations[pathology], ha='left', va='bottom',
                fontsize=6, fontweight='bold', color=color, rotation=0)

    # Add center lines at m=0.2 and Accuracy=0.8
    # ax.axvline(x=0.2, color='black', linestyle=':', linewidth=1, alpha=0.7)

    # Add quadrant labels
    # ax.text(0.15, 1, 'Low m\nHigh Acc', ha='center', va='center', fontsize=12,
    #         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    # ax.text(0.25, 1, 'High m\nHigh Acc', ha='center', va='center', fontsize=12,
    #         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    # ax.text(0.15, 0.2, 'Low m\nLow Acc', ha='center', va='center', fontsize=12,
    #         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    # ax.text(0.25, 0.2, 'High m\nLow Acc', ha='center', va='center', fontsize=12,
    #         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    x = np.linspace(0.001, 0.75, 100)
    ax.plot(x, np.full(x.shape, random_labeller_pd),
            color="steelblue", linestyle='dashdot', linewidth=3, alpha=0.7)

    random_baseline_handle, = plt.plot([], [], c='steelblue', linestyle='dashdot', linewidth=2,
                                       label='Random Baseline F1')

    # Create legend for PA levels
    pa_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color,
                                label=f'$p_d{pd_levels_symbol_maps.get(pa_level, '\\rightarrow')}${pa_level}')
                  for pa_level, color in color_map.items()]



    # Add human accuracy to legend
    human_handles = [plt.scatter([], [], c='gray', s=30, marker='o',
                                 label='Human Accuracy')]

    baseline_handle, = plt.plot([], [], c='gray',  linestyle = '--',
                                 label='Expected Accuracy')

    # Add Gemini marker to legend
    model_handles = [plt.scatter([], [], c='gray', s=30, marker='x',
                                 label=model_name)]

    all_handles = pa_handles + human_handles + model_handles + [baseline_handle, random_baseline_handle]+ pathology_handles
    ax.legend(handles=all_handles, loc='lower right', ncol=2, fontsize=8)

    ax.set_xlabel('Positive class ratio (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} Multidimensional Analysis with Expected Accuracy', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 0.72)
    ax.set_ylim(-0.01, 1.1)

    plt.tight_layout()
    return fig


def create_plots(json_scores_path: str, model_name: str, output_prefix: str):
    print("Creating Multi-Dimensional Analysis visualizations...")

    output_dir = Path(__file__).parent.parent.parent / "visualizations"
    output_dir.mkdir(exist_ok=True)

    # Create GP F1 visualization
    fig_gp_f1 = create_f1_vs_class_imbalance_plot(json_scores_path, model_name)
    fig_gp_f1.savefig(output_dir / f"multi_dimensional_analysis_{output_prefix}.png", dpi=400, bbox_inches='tight')
    plt.close(fig_gp_f1)
    print(f"F1 Multi-Dimensional Analysis visualization saved for {json_scores_path}")

    # Create GP Accuracy visualization
    fig_gp_acc = create_accuracy_vs_class_imbalance_plot(json_scores_path, model_name)
    fig_gp_acc.savefig(output_dir / f"multi_dimensional_analysis_{output_prefix}_accuracy.png", dpi=400,
                       bbox_inches='tight')
    plt.close(fig_gp_acc)
    print(f"Accuracy Multi-Dimensional Analysis visualization saved for {json_scores_path}")


def main():
    create_plots(
        Path(__file__).parent.parent.parent / "data_dir/checkpoints/gemini25pro/gemini25pro_scores_exclude_small_samples.json",
        "Gemini 2.5 Pro", "gemini25")
    create_plots(
        Path(__file__).parent.parent.parent / "data_dir/checkpoints/gemini30preview/gemini30preview_scores_exclude_small_samples.json",
        "Gemini 3.0 Preview", "gemini30")
    create_plots(Path(__file__).parent.parent.parent / "data_dir/checkpoints/gpt-5.1/gpt-5.1_scores_exclude_small_samples.json",
                 "GPT 5.1", "gpt51")


if __name__ == "__main__":
    main()
