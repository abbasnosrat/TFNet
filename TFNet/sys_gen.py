import control as c
import numpy as np
from control.matlab import lsim


class SystemGenerator:
    def __init__(self):
        """
        A random transfer function generator for testing TFNet.
        This object creates random transfer functions sampled from region C described in the paper.
        """
        self.u = np.zeros(30)
        self.u[:18] = 1
        self.z = c.TransferFunction.z
        self.p_size = [1, 1, 2, 2, 3]

    def first_order(self):
        p = np.random.random()
        p = (1 - 0.4) * p + 0.4  # to satisfy sampling assumption
        g = (1 - p)
        return g / (self.z - p)

    def type1(self):
        p = np.random.random()
        p = (1 - 0.4) * p + 0.4
        g = (1 - p)
        return g * 0.5 * (self.z + 1) / ((self.z - 1) * (self.z - p))

    def second_order(self):

        p = np.random.random(2)

        p = (1 - 0.4) * p + 0.4
        g = ((1 - p[0]) * (1 - p[1]))
        return g * 0.5 * (self.z + 1) / ((self.z - p[0]) * (self.z - p[1]))

    def second_order_complex(self):
        p = np.random.random(2)
        p[0] = (1 - 0.4) * p[0] + 0.4
        p[1] = (np.pi / 5) * p[1]
        pole = p[0] * np.cos(p[1]) + 1j * p[0] * np.sin(p[1])
        g = (1 - (2 * np.real(pole)) + (np.real(pole) ** 2 + np.imag(pole) ** 2))
        return g * 0.5 * (self.z + 1) / (
                self.z ** 2 - (2 * np.real(pole)) * self.z + (np.real(pole) ** 2 + np.imag(pole) ** 2))

    def type1_zero(self):
        allowed = False
        while not allowed:

            p = np.random.random(2)
            p[0] = (1 - 0.4) * p[0] + 0.4
            p[1] = (1.1 - 0.4) * p[1] + 0.4  # to have non-minimum phase systems
            if np.abs(p[1] - p[0]) > 0.02 or np.abs(p[1] - 1) > 0.02:
                allowed = True
        g = (1 - p[0])
        return g * (self.z - p[1]) / ((self.z - 1) * (self.z - p[0]))

    def second_order_zero(self):
        allowed = False
        while not allowed:
            p = np.random.random(3)
            p[0] = (1 - 0.4) * p[0] + 0.4
            p[1] = (1 - 0.4) * p[1] + 0.4
            p[2] = (1.1 - 0.4) * p[2] + 0.4
            if np.abs(p[2] - p[0]) > 0.02 or np.abs(p[2] - p[1]) > 0.02:
                allowed = True
        g = (((1 - p[0]) * (1 - p[1])) / (1 - p[2]))
        return g * (self.z - p[2]) / ((self.z - p[0]) * (self.z - p[1]))

    def second_order_zero_complex(self):
        p = np.random.random(3)
        p[0] = (1 - 0.4) * p[0] + 0.4
        p[1] = (np.pi / 5) * p[1]
        p[2] = (1.1 - 0.4) * p[2] + 0.4
        pole = p[0] * np.cos(p[1]) + 1j * p[0] * np.sin(p[1])
        g = (1 - (2 * np.real(pole)) + (np.real(pole) ** 2 + np.imag(pole) ** 2)) / (1 - p[2])
        return g * (self.z - p[2]) / (
                self.z ** 2 - (2 * np.real(pole)) * self.z + (np.real(pole) ** 2 + np.imag(pole) ** 2))

    def __call__(self, cls):
        """
        creates a random transfer function for any given class.
        {0: first order,
         1: type one,
         2: second order,
         3: type1 with zero,
         4: second order with zero}
        :param cls: class of the random transfer function
        :return: response of the random transfer function to the pulse signal, the transfer function object.
        """
        if cls == 0:
            Sys = self.first_order()
        elif cls == 1:
            Sys = self.type1()
        elif cls == 2:
            if bool(np.random.choice([0, 1], 1)):
                Sys = self.second_order()
            else:
                Sys = self.second_order_complex()

        elif cls == 3:
            Sys = self.type1_zero()
        elif cls == 4:
            if bool(np.random.choice([0, 1], 1)):
                Sys = self.second_order_zero()
            else:
                Sys = self.second_order_zero_complex()
        G = 10 * np.random.choice([1, -1], 1)
        Sys = Sys * G

        out, _, _ = lsim(Sys, self.u)
        N = np.linalg.norm(out)
        out = out+np.random.normal(0,0.01*N,out.shape)
        return out, Sys
