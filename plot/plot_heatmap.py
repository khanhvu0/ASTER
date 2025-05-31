import matplotlib.pyplot as plt
import numpy as np
import matplotlib

#Vertical colorbar heatmap
def vertical_cbar_heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar at bottom
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, location='bottom', shrink=0.4, pad=0.1)
    cbar.set_label(label=cbarlabel, labelpad=5)
    cbar.ax.tick_params(which="major", bottom=True, left=False, width=1, labelsize='large')
    cbar.ax.set_xlabel(cbarlabel, fontweight='semibold', fontsize='x-large')

    # Make colorbar tick labels bold using a more robust method
    for tick in cbar.ax.xaxis.get_major_ticks(): # Horizontal colorbar at bottom
        if tick.label1 is not None:
            tick.label1.set_fontweight('bold')
    # For completeness, if colorbar could be vertical
    for tick in cbar.ax.yaxis.get_major_ticks():
        if tick.label1 is not None:
            tick.label1.set_fontweight('bold')
        if tick.label2 is not None:
            tick.label2.set_fontweight('bold')

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Make x and y tick labels bold using a more robust method
    for tick in ax.xaxis.get_major_ticks():
        if tick.label2 is not None: # X-axis labels are on top
            tick.label2.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        if tick.label1 is not None: # Y-axis labels are on left
            tick.label1.set_fontweight('bold')

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="major", bottom=False, left=True, width=2, labelsize='large')
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            kw.update(fontsize='large')
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts    

# Data from the image
data_2d = np.array([
    [73.78, 75.74, 77.00, 77.52, 78.64, 78.88],
    [71.92, 74.20, 76.36, 76.52, 77.46, 77.94],
    [69.36, 72.38, 74.50, 75.84, 76.74, 77.18],
    [63.32, 69.86, 72.68, 73.12, 74.98, 75.44],
    [59.22, 66.50, 68.60, 69.92, 71.18, 71.88],
    [47.92, 59.50, 62.30, 63.46, 64.38, 65.24]
])


# Labels from the image
layers_skipped = [0, 1, 2, 3, 4, 5]
early_exit_threshold = [0.1, 0.3, 0.5, 0.6, 0.8, 0.9]

# Create subplots with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6))

# First heatmap
im1, cbar1 = vertical_cbar_heatmap(data_2d, layers_skipped, early_exit_threshold, ax=ax1,
                    cmap="YlGnBu", cbarlabel="Accuracy (%)")
texts1 = annotate_heatmap(im1, valfmt="{x:.2f}")
ax1.set_xlabel("Early exit threshold", fontweight='semibold', fontsize='large')
ax1.set_ylabel("Layers skipped", fontweight='semibold', fontsize='large')
# ax1.set_title("Dataset 1", fontweight='bold', fontsize='large', pad=10)

# Second heatmap
im2, cbar2 = vertical_cbar_heatmap(data_2d, layers_skipped, early_exit_threshold, ax=ax2,
                    cmap="YlGnBu", cbarlabel="Accuracy (%)")
texts2 = annotate_heatmap(im2, valfmt="{x:.2f}")
ax2.set_xlabel("Early exit threshold", fontweight='semibold', fontsize='large')
ax2.set_ylabel("Layers skipped", fontweight='semibold', fontsize='large')
# ax2.set_title("Dataset 2", fontweight='bold', fontsize='large', pad=10)

# Save the plot to a file
plt.savefig(fname="sdt_imagenet100_heatmap.png", dpi=300, format='png', 
            bbox_inches='tight', pad_inches=0.2,
            facecolor='auto', edgecolor='auto')

# Also show the plot
plt.show()
