#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def load_groundtruth_data():
    base_path = Path(__file__).parent.parent.parent / "data_dir" / "chexpert" / "groundtruth"
    
    all_data = []
    for file_path in base_path.glob("*.csv"):
        df = pd.read_csv(file_path)
        df['patient'] = df['Study'].str.extract(r'patient(\d+)').astype(int)
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

def load_gemini25_data():
    """Load Gemini-2.5-pro predictions only"""
    checkpoint_path = Path(__file__).parent.parent.parent / "data_dir" / "checkpoints" / "gemini25pro" / "checkpoint.json"
    
    with open(checkpoint_path, 'r') as f:
        data = json.load(f)
    
    model_predictions = []
    for entry in data:
        patient_id = int(entry['patient'].replace('patient', ''))
        predictions = entry['model_response']
        
        model_predictions.append({
            'patient': patient_id,
            'has_cardiomegaly': predictions.get('has_cardiomegaly', False),
            'has_lung_opacity': predictions.get('has_lung_opacity', False),
            'has_edema': predictions.get('has_edema', False),
            'has_consolidation': predictions.get('has_consolidation', False),
            'has_pneumonia': predictions.get('has_pneumonia', False)
        })
    
    return pd.DataFrame(model_predictions)

def create_pentagon_gemini25_focus():
    df = load_groundtruth_data()
    gemini_df = load_gemini25_data()
    
    conditions = ['Cardiomegaly', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia']
    condition_keys = ['has_cardiomegaly', 'has_lung_opacity', 'has_edema', 'has_consolidation', 'has_pneumonia']
    
    # Pentagon corners
    angles = np.linspace(0, 2*np.pi, 6)[:-1] - np.pi/2
    
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Draw pentagon rings for all 6 levels with correct colors
    ring_colors = ['black', '#888888', '#cccccc', '#cccccc', '#888888', 'black']  # 0/5, 1/5, 2/5, 3/5, 4/5, 5/5
    
    for level in range(6):  # 0/5 to 5/5
        radius = (level + 1) / 7  # Adjust scaling to fit 6 levels
        pentagon_x = [radius * np.cos(angle) for angle in angles] + [radius * np.cos(angles[0])]
        pentagon_y = [radius * np.sin(angle) for angle in angles] + [radius * np.sin(angles[0])]
        ax.plot(pentagon_x, pentagon_y, color=ring_colors[level], alpha=0.6, linewidth=2)
    
    # For each condition, calculate Gemini agreement at each vote level
    for cond_idx, (condition, condition_key) in enumerate(zip(conditions, condition_keys)):
        
        for vote_level in range(6):  # 0/5 to 5/5
            # Get patients with this specific vote level
            patients_at_level = []
            for patient in df['patient'].unique():
                patient_data = df[df['patient'] == patient][condition].dropna()
                if len(patient_data) == 5:
                    positive_votes = (patient_data == 1).sum()
                    if positive_votes == vote_level:
                        patients_at_level.append(patient)
            
            if len(patients_at_level) > 0:
                # Calculate Gemini agreement for patients at this vote level
                agree_count = 0
                total_count = 0
                
                for patient in patients_at_level:
                    if patient in gemini_df['patient'].values:
                        # Human majority at this level
                        human_vote = vote_level >= 3  # 3+ votes = positive majority
                        # Gemini prediction
                        gemini_vote = gemini_df[gemini_df['patient'] == patient][condition_key].iloc[0]
                        
                        total_count += 1
                        if human_vote == gemini_vote:
                            agree_count += 1
                
                if total_count > 0:
                    agreement_rate = agree_count / total_count
                    
                    # Position at the vote level ring
                    radius = (vote_level + 1) / 7  # Match the ring scaling
                    x = radius * np.cos(angles[cond_idx])
                    y = radius * np.sin(angles[cond_idx])
                    
                    # Circle size based on sample count
                    circle_size = max(50, total_count * 10)
                    
                    # Color based on agreement rate: three levels
                    condition_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
                    condition_light = ['#ffb3b3', '#b3d9ff', '#b3ffb3', '#e6b3ff', '#ffcc99']  # Lighter versions
                    
                    if agreement_rate > 0.8:  # High agreement
                        circle_color = condition_colors[cond_idx]
                    elif agreement_rate >= 0.6:  # Medium agreement
                        circle_color = condition_light[cond_idx]  # Lighter condition color
                    else:  # Low agreement < 0.6
                        circle_color = '#cccccc'  # Light gray
                    
                    ax.scatter(x, y, s=circle_size, alpha=0.8, color=circle_color, 
                              edgecolors='black', linewidth=1)
    
    # Label conditions
    condition_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    for cond_idx, condition in enumerate(conditions):
        label_x = 1.05 * np.cos(angles[cond_idx])
        label_y = 1.05 * np.sin(angles[cond_idx])
        
        if condition == 'Lung Opacity':
            label_y -= 0.1
            
        ax.text(label_x, label_y, condition, ha='center', va='center',
               fontsize=12, fontweight='bold', color=condition_colors[cond_idx])
    
    # Center point
    ax.scatter(0, 0, c='black', s=50, marker='o')
    
    # Vote level legend
    ring_legend_elements = []
    ring_labels = ['0/5', '1/5', '2/5', '3/5', '4/5', '5/5']
    for i, (color, label) in enumerate(zip(ring_colors, ring_labels)):
        ring_legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=3, 
                                              label=f'{label} votes'))
    
    # Only keep vote level legend
    ring_legend = ax.legend(handles=ring_legend_elements, loc='upper left', 
                           bbox_to_anchor=(0, 1), title='Vote Levels')
    ax.add_artist(ring_legend)
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.set_title('Gemini-2.5-pro Agreement with Human Experts by Vote Level\nFull color = high (>80%) | Light color = medium (60-80%) | Gray = low (<60%)',
                fontsize=14, fontweight='bold', pad=30)
    
    return fig

def main():
    print("Creating Gemini-2.5-pro focused pentagon visualization...")
    fig = create_pentagon_gemini25_focus()
    
    output_dir = Path(__file__).parent.parent.parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "pentagon_gemini25_focus.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("✓ Gemini-2.5-pro focused pentagon visualization saved")

if __name__ == "__main__":
    main()
