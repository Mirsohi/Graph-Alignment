"""Shared figure style (round 13): one font family (serif) for every figure, and
figure widths equal to their printed widths, so that all text prints at >= 7 pt."""
TEXTWIDTH = 6.75   # inches, the paper's text width
STYLE = {'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
         'legend.frameon': False, 'pdf.fonttype': 42, 'mathtext.fontset': 'dejavuserif'}
