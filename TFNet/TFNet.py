import importlib.resources as pkg_resources
import os

import control as c
import numpy as np
import pandas as pd
import torch
from control.matlab import lsim
from tqdm.auto import trange

from TFNet.network import Resnet


class TFNet:
    def __init__(self, std=0.01):
        """
        TFNet object takes the response of the system to a pulse excitation and identifies both
        the structure and parameters of its transfer function.
        :param std: approximate standard deviation of the measurement noise divided by the norm of the response. default=0.01

        """
        self.DIR = pkg_resources.files("TFNet")

        self.configs = {"layers": [[1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]],
                        "output_size": [1, 1, 2, 2, 3],

                        "weights": [
                            os.path.join("weights", "first_order", f"first_order_{std}.pt"),
                            os.path.join("weights", "type1", f"type1_{std}.pt"),
                            os.path.join("weights", "second_order", f"second_order_{std}.pt"),
                            os.path.join("weights", "type1_zero", f"type1_zero_{std}.pt"),
                            os.path.join("weights", "second_order_zero", f"second_order_zero_{std}.pt")]}
        self.configs = pd.DataFrame(self.configs)
        self.u = np.zeros(30)
        self.u[:18] = 1
        self.z = c.TransferFunction.z
        self.parameters = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def preprocess_input(self, X):
        """
        takes the response signal, normalizes it and reshapes it such that it can be fed to the CNN
        :param X: The response signal. The shape must be [-1,30]
        :return: The processed signal, norm of the signal
        """
        N_true = np.linalg.norm(X, axis=-1)
        N_true = N_true.reshape(-1, 1)
        X = X / np.repeat(N_true, X.shape[-1], axis=1)
        X = torch.from_numpy(X).detach()
        X = X.float().to(self.device)
        X = X.reshape([-1, 1, X.shape[-1]])
        return X, N_true.reshape(-1)

    def estimation_phase(self, X):
        """
        This method is the estimation stage described in the paper. The response is fed to the corresponding
        CNN of each class and the parameters are estimated.
        :param X: The processed response.
        :return: parameters of the transfer function corresponding to each class.
        """

        p = torch.zeros([X.shape[0], 5, 3])
        for i in range(5):
            cls = self.configs.iloc[i]
            model = Resnet(cls.layers, 1, cls.output_size).to(self.device)

            chp = torch.load(os.path.join(self.DIR, cls.weights))
            model.load_state_dict(chp)
            model.eval()
            with torch.no_grad():
                Pi = model(X)
                p[:, i, :cls.output_size] = Pi

        return p.detach().cpu().numpy()

    def detection_phase(self, X, p, N_true):
        """
        This method is the detection stage described in the paper. The parameters from the previous stage
        are given to this stage to simulate the output of their corresponding transfer function. Afterward,
        the best structure is selected from argmin of the euclidian distance between the response signal and
        the simulated outputs. The gain is evaluated from N_true and the norm of the simulated outputs.
        :param X: The processed response signal
        :param p: Parameters estimated at the estimation stage.
        :param N_true: Norm of the response.
        :return: The identified transfer function object.
        """
        X = X.reshape(1, 30)
        gain = [1, -1]
        predictions = np.zeros([10, 30])
        norms = []

        systems = self.tf(p)
        for cls in range(5):
            out, _, _ = lsim(systems[cls], self.u)
            N = np.linalg.norm(out)
            out = out / N
            norms.append(N)
            predictions[2 * cls, :] = out
            predictions[2 * cls + 1] = -out
        errs = np.linalg.norm(X - predictions, axis=-1)
        # errs = X @ predictions.T

        i_star = np.argmin(errs, axis=-1)

        cls_star = int(i_star / 2)

        gain_sign = gain[int(i_star % 2)]
        k = gain_sign * N_true / norms[cls_star]

        Sys = k * systems[cls_star]

        return Sys

    def tf(self, p):
        """
        creates transfer function objects from the given parameters.
        :param p: parameters estimated at the estimation stage.
        :return: A list of transfer function objects corresponding to each class.
        """

        return [p[0, 0] / (self.z - p[0, 0]),

                (1 - p[1, 0]) * 0.5 * (self.z + 1) / ((self.z - p[1, 0]) * (self.z - 1)),

                (1 ** 2 + p[2, 0] * 1 + p[2, 1]) * 0.5 * (self.z + 1) / (self.z ** 2 + p[2, 0] * self.z + p[2, 1]),

                np.abs((1 - p[3, 0])) * (self.z - p[3, 1]) / ((self.z - 1) * (self.z - p[3, 0])),

                ((1 ** 2 + p[4, 0] * 1 + p[4, 1]) / (1 - p[4, 2])) * (self.z - p[4, 2]) / (
                        self.z ** 2 + p[4, 0] * self.z + p[4, 1])]

    def __call__(self, X):
        """
        This method identifies a transfer function model of the system from its response to the pulse signal by
        preprocessing it, estimating the model parameters via the estimation stage, and identifying the best
        structure along with evaluation of the model gain via the detection stage.

        :param X: The response signal. The shape must be [-1,30]
        :return: The identified transfer function object.
        """

        systems = []
        X = X.reshape([-1, 30])
        X, N_true = self.preprocess_input(X)
        p = self.estimation_phase(X)
        X = X.detach().cpu().numpy()
        for i in trange(X.shape[0],leave=True):
            Sys = self.detection_phase(X[i], p[i], N_true[i])
            systems.append(Sys)

        return systems
