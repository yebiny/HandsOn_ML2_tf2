# TensorFlow ≥2.0 is required
import tensorflow as tf

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
mpl.rcParams["figure.figsize"] = (12,4)

import tensorflow as tf
from tensorflow.keras import models, layers


n_steps=50
def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

def plot_series(idx, series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$"):
    plt.plot(series[idx], ".-")
    if y is not None:
        plt.plot(n_steps, y[idx], "bx", markersize=10)
    if y_pred is not None:
        plt.plot(n_steps, y_pred[idx], "ro")
    plt.title("input : %i"%idx)
    plt.grid(True)
    #plt.yticks([])
    if x_label!=None:
        plt.xlabel(x_label, fontsize=16)
    if y_label!=None:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps + 1, -1, 1])
    

def plot_multiple_forecasts(idx, X, Y, Y_pred, y_label="$x(t)$"):
    n_steps = len(X[idx])
    ahead = len(Y[idx])
    plot_series(idx, X, y_label=y_label)
    plt.plot(np.arange(n_steps, n_steps + ahead), Y[idx], "bx-", label="Actual")
    if Y_pred is not None:
        plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[idx], "r.-", label="Forecast", markersize=10)
    plt.axis([0, n_steps + ahead, -1, 1])
    plt.legend(fontsize=14)
    
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
    return series[..., np.newaxis].astype(np.float32)
