#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from matplotlib.patches import Ellipse

pd_levels = [0.6, 0.8, 1.0, 'All']
color_map = {0.6: '#d62728', 0.8: '#ff7f0e', 1.0: '#2ca02c', 'All': 'gray'}
pd_levels_symbol_maps = {
    "All":"="
}
def load_data(json_path):
    """Load data from JSON checkpoint"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def create_single_model_plot(json_path, model_name):
    """Create plot for a single model"""
    df = load_data(json_path)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
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
        
        # Create ellipse that shows the actual range
        width = (m_max - m_min)
        height = (delta_max - delta_min)
        ellipse = Ellipse((avg_m, avg_delta), width, height, 
                        facecolor=color, alpha=0.2, edgecolor=color, linewidth=1)
        ax.add_patch(ellipse)
        
        # Plot center point
        ax.scatter(avg_m, avg_delta, c=color, s=total_samples/5, alpha=0.9, 
                  marker='o', edgecolors='black', linewidth=1, zorder=5)
        
        # Add sample count label
        ax.text(avg_m + 0.01, avg_delta, f'n={total_samples}', 
               fontsize=8, ha='left', va='center')
    
    # Add horizontal line at y=0 (no difference)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No difference')
    
    # Create legend for PA levels
    pa_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color,
                                label=f'$p_d{pd_levels_symbol_maps.get(pa_level, '\\rightarrow')}${pa_level}')
                  for pa_level, color in color_map.items()]
    
    ax.legend(handles=pa_handles, loc='upper left', fontsize=10)
    
    ax.set_xlabel('Average Positive class ratio (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Δ F1 (Human - Model)', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name}: Performance Delta by PD Level\n(Circle size = sample count, Ellipse = range)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_single_model_accuracy_plot(json_path, model_name):
    """Create accuracy plot for a single model"""
    df = load_data(json_path)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Group by PA level and calculate averages
    for pa_level in pd_levels:
        pa_data = df[df['PA'] == pa_level]
        
        if len(pa_data) == 0:
            continue
            
        # Calculate delta (model - human) for accuracy
        deltas = pa_data['agg_human - model_mean_accuracy']
        avg_delta = deltas.mean()
        delta_min, delta_max = deltas.min(), deltas.max()
        m_values = pa_data['m'].astype(float)
        avg_m = m_values.mean()
        m_min, m_max = m_values.min(), m_values.max()
        total_samples = int(pa_data['S'].mean())
        
        # Get color for this PA level
        color = color_map[pa_level]
        
        # Create ellipse that shows the actual range
        width = (m_max - m_min)
        height = (delta_max - delta_min)
        ellipse = Ellipse((avg_m, avg_delta), width, height, 
                        facecolor=color, alpha=0.2, edgecolor=color, linewidth=1)
        ax.add_patch(ellipse)
        
        # Plot center point
        ax.scatter(avg_m, avg_delta, c=color, s=total_samples/5, alpha=0.9, 
                  marker='o', edgecolors='black', linewidth=1, zorder=5)
        
        # Add sample count label
        ax.text(avg_m + 0.01, avg_delta, f'n={total_samples}',
               fontsize=8, ha='left', va='center')
    
    # Add horizontal line at y=0 (no difference)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No difference')
    
    # Create legend for PA levels
    pa_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color,
                                label=f'$p_d{pd_levels_symbol_maps.get(pa_level, '\\rightarrow')}${pa_level}')
                  for pa_level, color in color_map.items()]
    
    ax.legend(handles=pa_handles, loc='upper left', fontsize=10)
    
    ax.set_xlabel('Average Positive class ratio (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Δ Accuracy (Human - Model)', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name}: Accuracy Delta by PD Level\n(Circle size = sample count, Ellipse = range)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    print("Creating individual model Delta F1 and Accuracy Analysis plots...")
    
    output_dir = Path(__file__).parent.parent.parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    models = [
        ('Gemini 2.5 Pro', Path(__file__).parent.parent.parent / "data_dir/checkpoints/gemini25pro/gemini25pro_scores_exclude_small_samples.json", "gemini25"),
        ('Gemini 3.0 Preview', Path(__file__).parent.parent.parent / "data_dir/checkpoints/gemini30preview/gemini30preview_scores_exclude_small_samples.json", "gemini30"),
        ('GPT 5.1', Path(__file__).parent.parent.parent / "data_dir/checkpoints/gpt-5.1/gpt-5.1_scores_exclude_small_samples.json", "gpt51")
    ]
    
    for model_name, json_path, prefix in models:
        # F1 plots
        fig = create_single_model_plot(json_path, model_name)
        fig.savefig(output_dir / f"single_model_delta_f1_{prefix}.png", dpi=400, bbox_inches='tight')
        plt.close(fig)
        print(f"Single model F1 plot saved for {model_name}")
        
        # Accuracy plots
        fig = create_single_model_accuracy_plot(json_path, model_name)
        fig.savefig(output_dir / f"single_model_delta_accuracy_{prefix}.png", dpi=400, bbox_inches='tight')
        plt.close(fig)
        print(f"Single model accuracy plot saved for {model_name}")

if __name__ == "__main__":
    main()
