import matplotlib.pyplot as plt
import numpy as np

# Data from terminal output
class_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
accuracy_data = [76.00, 82.00, 46.00, 56.00, 69.00, 68.00, 76.00, 69.00, 76.00, 91.00]
avg_timestep_data = [3.93, 3.89, 4.38, 5.23, 4.27, 4.17, 3.72, 3.66, 3.44, 3.20]


# Plot settings
bar_width = 0.35
x = np.arange(len(class_ids)) # Use numpy arange for positioning bars

# Define hatch patterns
accuracy_hatch = '//'
timestep_hatch = '\\'

# Custom colors
accuracy_color = '#FAEDCD'  
timestep_color = '#D4A373' 

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 4)) # Adjusted figsize for better readability

# Bar plots for accuracy
ax1.bar(x - bar_width/2, accuracy_data, width=bar_width, label='Accuracy (%)', color=accuracy_color, hatch=accuracy_hatch, zorder=3, edgecolor='black')
ax1.set_ylabel("Accuracy (%)", fontsize=14, fontweight='bold', color='black')
ax1.tick_params(axis='y', labelcolor='black', labelsize=12, width=2)
for label in ax1.get_yticklabels():
    label.set_weight('bold')
ax1.set_ylim(0, 100) # Accuracy is a percentage

# Secondary Y-axis for average timestep
ax2 = ax1.twinx()
ax2.bar(x + bar_width/2, avg_timestep_data, width=bar_width, label='Avg Timestep', color=timestep_color, hatch=timestep_hatch, zorder=3, edgecolor='black')
ax2.set_ylabel("Average Timestep", fontsize=14, fontweight='bold', color='black')
ax2.tick_params(axis='y', labelcolor='black', labelsize=12, width=2)
for label in ax2.get_yticklabels():
    label.set_weight('bold')
# Determine a reasonable upper limit for avg_timestep y-axis, e.g., max value + a small margin
ax2.set_ylim(0, max(avg_timestep_data) * 1.15 if avg_timestep_data else 10)


# Primary X-axis labels and ticks (common for both y-axes)
# ax1.set_xlabel("Class Name", fontsize=14, fontweight='bold')
ax1.set_xticks(ticks=x)
ax1.set_xticklabels(class_names, fontsize=14, fontweight='bold', rotation=45, ha='right')
ax1.tick_params(axis='x', labelsize=10, width=2)
for label in ax1.get_xticklabels():
    label.set_weight('bold')

ax1.grid(axis='y', linestyle='--', linewidth=0.5, zorder=0, alpha=0.7) # Grid for primary y-axis
ax1.set_axisbelow(True)


# Thicker spines
for spine_pos in ['left', 'bottom']:
    ax1.spines[spine_pos].set_linewidth(2)
    ax1.spines[spine_pos].set_color('black')

for spine_pos in ['right', 'top']:
    ax2.spines[spine_pos].set_linewidth(2)
    ax2.spines[spine_pos].set_color('black') # Keep black or match respective axis color

ax1.spines['left'].set_color('black')
ax2.spines['right'].set_color('black')
ax1.spines['top'].set_visible(False) # Hide top spine for cleaner look with twinx
ax2.spines['top'].set_visible(False)


# Combined Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
legend = fig.legend(lines1 + lines2, labels1 + labels2, loc='upper center', ncol=2, fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.98)) # Position legend at top center inside the plot
for text in legend.get_texts():
    text.set_weight('bold')

# Tight layout and save
plt.tight_layout() # Adjust rect if legend needs more space or to prevent overlap rect=[0, 0, 1, 0.95]
plt.savefig('early_exit_per_class_stats.svg', format='svg')
plt.show()