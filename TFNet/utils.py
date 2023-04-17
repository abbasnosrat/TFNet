import matplotlib.pyplot as plt
import numpy as np
from control.matlab import lsim
from matplotlib.style import use


def plot_results(sys_true_list, sys_pred, responses):
    use("seaborn")
    u = np.zeros(30)
    u[:18] = 1
    plt.figure(figsize=[13, 6])
    for i in range(len(sys_pred)):
        plt.subplot(1, len(sys_pred), i + 1)
        y, _, _ = lsim(sys_pred[i], u)
        plt.plot(responses[i, :], label="True")
        plt.plot(y, "--", label="TFNet")
        plt.title(f"true: {sys_true_list[i]}, TFNet: {sys_pred[i]}", fontsize=10)
        plt.grid("on")
    plt.legend()
