import matplotlib.pyplot as plt

# Data from the table
thresholds = [
    "0.8", "0.9", "0.95", "0.99", "0.999",
    "0.9995", "0.9999", "0.99999", "0.999999", "0.9999999"
]
timesteps_reduction = [
    13.93, 13.41, 12.84, 12.01, 10.92,
    10.51, 9.86, 8.88, 8.14, 7.3
]
accuracy = [
    63.7, 66.6, 68.3, 70.9, 74.0,
    74.7, 76.0, 76.7, 77.2, 77.4
]

fig, ax1 = plt.subplots(figsize=(10, 6))

# X values as categorical positions
x = list(range(len(thresholds)))

# Left Y-axis: timesteps reduction
color1 = 'tab:blue'
ax1.set_xlabel('Early exit threshold')
ax1.set_ylabel('Timesteps reduction', color=color1)
ax1.plot(x, timesteps_reduction, marker='o', color=color1, label='Timesteps reduction')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xticks(x)
ax1.set_xticklabels(thresholds)

# Right Y-axis: accuracy
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Accuracy (%)', color=color2)
ax2.plot(x, accuracy, marker='s', color=color2, label='Accuracy')
ax2.tick_params(axis='y', labelcolor=color2)

# Dotted horizontal baseline
ax1.axhline(y=16, color='gray', linestyle=':', linewidth=2)
ax2.axhline(y=78.8, color='gray', linestyle=':', linewidth=2)
plt.text(
    x[-1], 16.2, "baseline: 16 timesteps, 78.8% accuracy",
    ha='right', va='bottom', fontsize=12, color='gray'
)

# Legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower left')

plt.title('Early Exit Threshold vs Timesteps Reduction and Accuracy')
# plt.tight_layout()
# plt.show()
plt.savefig("early_exit_plot.png", dpi=300)
plt.close()