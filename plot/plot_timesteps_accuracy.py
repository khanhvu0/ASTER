import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Data
# data = {
#     "Early exit threshold": [0.8, 0.9, 0.95, 0.99, 0.999, 0.9995, 0.9999, 0.99999, 0.999999, 0.9999999],
#     "Timesteps": [13.66, 13.18, 12.9, 12.5, 11.9, 11.57, 11.23, 10.92, 10.19, 9.85],
#     "Accuracy": [80.56, 84.03, 86.81, 90.28, 94.1, 94.79, 95.83, 95.83, 97.22, 97.57]
# }

data = {
    "Early exit threshold": [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999],
    "Timesteps": [2.27, 2.17, 2.07, 2, 1.92, 1.82, 1.7, 1.38, 1.17, 0.93],
    "Accuracy": [73.78, 75.74, 77, 77.52, 78.14, 78.64, 78.64, 79.16, 79.26, 79.24]
}

df = pd.DataFrame(data)

# Scientific-style X-axis labels corresponding to (1 - threshold)
# sci_labels = [
#     r"$2\times10^{-1}$",
#     r"$10^{-1}$",
#     r"$5\times10^{-2}$",
#     r"$10^{-2}$",
#     r"$10^{-3}$",
#     r"$5\times10^{-4}$",
#     r"$10^{-4}$",
#     r"$10^{-5}$",
#     r"$10^{-6}$",
#     r"$10^{-7}$"
# ]
sci_labels = [
    r"$9\times10^{-1}$",
    r"$7\times10^{-1}$",
    r"$5\times10^{-1}$",
    r"$4\times10^{-1}$",
    r"$3\times10^{-1}$",
    r"$2\times10^{-1}$",
    r"$10^{-1}$",
    r"$10^{-2}$",
    r"$10^{-3}$",
    r"$10^{-4}$",
]

# Plot
fig, ax1 = plt.subplots(figsize=(10, 4))

# Primary Y-axis: Timesteps reduction
ax1_color = '#577399'
ax1.set_xlabel('(1 - Early exit threshold)', fontsize=18, weight='bold')
ax1.set_ylabel('Timesteps Reduction', color=ax1_color, fontsize=18, weight='bold')
ax1.plot(sci_labels, df["Timesteps"], color=ax1_color, marker='s', markersize=10, linewidth=3.0, label='Timesteps Reduction')
ax1.tick_params(axis='y', labelcolor=ax1_color, labelsize=14, width=2, color=ax1_color)
ax1.tick_params(axis='x', labelsize=16, width=2)
# ax1.yaxis.set_major_locator(ticker.MultipleLocator(4))  # Tick step size = 4
ax1.set_ylim(0, 4)  # Set Y-axis limit to show tick up to 16
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_axisbelow(True)

# Secondary Y-axis: Accuracy
ax2 = ax1.twinx()
ax2_color = '#66c2a5'
ax2.set_ylabel('Accuracy (%)', color=ax2_color, fontsize=18, weight='bold')
ax2.plot(sci_labels, df["Accuracy"], color=ax2_color, marker='^', markersize=10, linewidth=3.0, label='Accuracy')
ax2.tick_params(axis='y', labelcolor=ax2_color, labelsize=14, width=2, color=ax2_color)
ax2.set_ylim(73, 80)

ax2.axhline(y=79.26, color='gray', linestyle='--', linewidth=2)
ax2.annotate('Baseline Accuracy',
             xy=(0.02, 79), xycoords=('axes fraction', 'data'),
             xytext=(10, -10), textcoords='offset points',
             fontsize=14, weight='bold', color='gray')
    
# Thicker spines for better readability
for spine in ['left', 'bottom']:
    ax1.spines[spine].set_linewidth(2)
for spine in ['right', 'top']:
    ax2.spines[spine].set_linewidth(2)
# ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_color(ax1_color)
ax2.spines['right'].set_color(ax2_color)

# For ax1
ax1.tick_params(axis='both', labelsize=14, width=2)
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_weight('bold')

# For ax2
ax2.tick_params(axis='y', labelsize=14, width=2)
for label in ax2.get_yticklabels():
    label.set_weight('bold')

# Title and Legend
plt.title('SDT-8-512 on Imagenet100', fontsize=15, weight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(1, 0.65), fontsize=14)
for text in legend.get_texts():
    text.set_weight('bold')
plt.tight_layout()
plt.savefig("imagenet100_timestep_vs_accuracy.svg", format='svg')
plt.show()