import os
import matplotlib.pyplot as plt


from .file import *

def draw_line(x, y, filepath, xlabel=None, ylabel=None, show=False):

    fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(x, y)
    if xlabel is not None and ylabel is not None:
        ax.set_title(ylabel + ' vs. ' + xlabel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.grid(which='minor', axis='both')
    fig.savefig(filepath)

    if show:
        plt.show()
    return filepath