#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_groundtruth_data():
    base_path = Path(__file__).parent.parent.parent / "data_dir" / "chexpert" / "groundtruth"
    
    all_data = []
    for file_path in base_path.glob("*.csv"):
        df = pd.read_csv(file_path)
        df['patient'] = df['Study'].str.extract(r'patient(\d+)').astype(int)
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

def create_pentagon_radial_levels():
    df = load_groundtruth_data()
    conditions = ['Cardiomegaly', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia']
    
    # Pentagon corners
    angles = np.linspace(0, 2*np.pi, 6)[:-1] - np.pi/2
    
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Draw concentric pentagons for each level
    colors = ['black', '#666666', '#cccccc', '#cccccc', '#666666', 'black']  # 0/5, 1/5, 2/5, 3/5, 4/5, 5/5
    
    for level in range(6):  # 0/5 to 5/5
        radius = (level + 0.9) / 6  # Start from center, go outward
        pentagon_x = [radius * np.cos(angle) for angle in angles] + [radius * np.cos(angles[0])]
        pentagon_y = [radius * np.sin(angle) for angle in angles] + [radius * np.sin(angles[0])]
        ax.plot(pentagon_x, pentagon_y, color=colors[level], alpha=0.8, linewidth=2, 
               label=f'{level}/5 votes')
    
    # Count samples at each condition-vote combination
    condition_colors = ['bisque', 'paleturquoise', 'lightsalmon', 'thistle', 'lightpink']  # Different color per condition
    font_color = {
        'bisque': 'darkorange',
        "paleturquoise":"teal",
        "lightsalmon":"tomato",
        "thistle": "darkviolet",
       "lightpink": "crimson"
    }
    for cond_idx, condition in enumerate(conditions):
        vote_counts = {}  # vote_level -> count

        NUM_ANNOTATIONS = 5
        for patient in df['patient'].unique():
            patient_data = df[df['patient'] == patient][condition].dropna()
            if len(patient_data) == NUM_ANNOTATIONS:
                positive_votes = (patient_data == 1).sum()
                vote_counts[positive_votes] = vote_counts.get(positive_votes, 0) + 1
        
        # Plot circles sized by sample count
        for votes, count in vote_counts.items():

            radius = (votes + 0.9) / 6
            x = radius * np.cos(angles[cond_idx])
            y = radius * np.sin(angles[cond_idx])
            
            # Circle size proportional to sample count with minimum size
            circle_size = max(50, count * 10)  # Minimum size 50 for visibility
            
            #condition color 
            circle_color = condition_colors[cond_idx]
            
            ax.scatter(x, y, s=circle_size, alpha=0.7, color=circle_color, 
                      edgecolors=circle_color, linewidth=1)
            
            # Add count label with white text
            ax.text(x, y, str(round(count * NUM_ANNOTATIONS / len(df['patient']), 3)), ha='left', va='bottom', fontsize=10,
                    fontweight='bold', color=font_color.get(circle_color,'darkblue'))
    
    # Label conditions at outermost corners
    for cond_idx, condition in enumerate(conditions):
        label_x = 1.05 * np.cos(angles[cond_idx])
        label_y = 1.05 * np.sin(angles[cond_idx])
        
        # # Adjust Lung Opacity position slightly down
        # if condition == 'Lung Opacity':
        #     label_y -= 0.1
            
        ax.text(label_x, label_y, condition, ha='center', va='center',
               fontsize=12, fontweight='bold', color=font_color.get(condition_colors[cond_idx],'darkblue'))
    
    # Center point
    #ax.scatter(0, 0, c='gray', s=25, marker='o')
    # ax.text(0, -0.15, 'Center\n0/5', ha='center', va='top', fontsize=10)
    
    # Create legend for vote levels (pentagon rings)
    legend_elements = []
    vote_labels = ['0/5 (All Negative)', '1/5', '2/5', '3/5', '4/5', '5/5 (All Positive)']
    for level in range(6):
        legend_elements.append(plt.Line2D([0], [0], color=colors[level], linewidth=3, 
                                        label=vote_labels[level]))
    ax.legend(handles=legend_elements, loc='upper right',  bbox_to_anchor=(1.2, 0.7),
             title='Agreement Levels', fontsize=14)
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("")
    
    # ax.set_title('Pentagon Agreement Visualization\nCircle size = sample count | Numbers show exact counts',
    #             fontsize=14, fontweight='bold', pad=30)
    
    return fig

def main():
    print("Creating pentagon with radial levels visualization...")
    fig = create_pentagon_radial_levels()
    
    output_dir = Path(__file__).parent.parent.parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "pentagon_radial_levels.png", dpi=600, bbox_inches='tight')
    plt.close(fig)
    
    print("✓ Pentagon radial levels visualization saved")

if __name__ == "__main__":
    main()
