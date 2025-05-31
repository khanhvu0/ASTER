import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the model folders and their labels
model_folders = [
    ('6-512-og', '6-512 Original'),
    ('8-512-og', '8-512 Original'),
    ('6-512-lin', '6-512 Linear'),
    ('8-512-lin', '8-512 Linear'),
    ('6-512-attn', '6-512 Attention'),
    ('8-512-attn', '8-512 Attention'),
    ('6-512-skip', '6-512 Skip'),
    ('8-512-skip', '8-512 Skip')
]

# Split models into 6-512 and 8-512 groups
models_6_512 = [(f, l) for f, l in model_folders if f.startswith('6-512')]
models_8_512 = [(f, l) for f, l in model_folders if f.startswith('8-512')]

# Define colors and line styles for better visualization
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
line_styles = ['-', '--', ':', '-.']

def plot_accuracy_curves(models, title, colors, ax):
    """Plot accuracy curves for a group of models"""
    handles = []
    labels = []
    
    for (folder, label), color, line_style in zip(models, colors, line_styles):
        file_path = os.path.join('output/train', folder, 'summary.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Plot the accuracy curve with both color and line style
            line = ax.plot(df['epoch'], df['eval_top1'], 
                    color=color, linestyle=line_style, linewidth=2)
            
            # Find the best accuracy
            best_acc = df['eval_top1'].max()
            
            # Store the line and label with best accuracy
            handles.append(line[0])
            labels.append(f'{label} (Best: {best_acc:.1f}%)')
    
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(handles, labels, loc='lower right', fontsize=10)

def plot_loss_curves(models, title, colors, ax):
    """Plot loss curves for a group of models"""
    for (folder, label), color, line_style in zip(models, colors, line_styles):
        file_path = os.path.join('output/train', folder, 'summary.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            ax.plot(df['epoch'], df['train_loss'], 
                    label=label, color=color, linestyle=line_style, linewidth=2)
    
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=10)

# Create figure for accuracy curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
plot_accuracy_curves(models_6_512, '6-512 Model Variations', colors, ax1)
plot_accuracy_curves(models_8_512, '8-512 Model Variations', colors, ax2)
plt.tight_layout()
plt.savefig('accuracy_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# Create figure for loss curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
plot_loss_curves(models_6_512, '6-512 Training Loss', colors, ax1)
plot_loss_curves(models_8_512, '8-512 Training Loss', colors, ax2)
plt.tight_layout()
plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
plt.close() 