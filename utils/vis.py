import os
import torch
import time

import numpy as np
import matplotlib.pyplot as plt

class dynamic_pic():
    r"""need to plt.show() at end of the code,
        examples:
        >>> monitor = dynamic_pic("Dynamic representation", xlabel="x", ylabel="y")
        >>> for i in range(100):
        >>>     monitor(i*0.1, np.sin(i*0.1))
        >>>     plt.show()
    """
    def __init__(self,title, xlabel="", ylabel="",  style="", pause=0.001, color="orange"):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.style = style
        self.pause = pause
        self.color = color

        self.xlist = []
        self.ylist = []

        self._set_draw()

    def _set_draw(self):
        assert len(self.xlist) == len(self.ylist), "xlist must have the same length as ylist"
        plt.figure()
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)

    def __call__(self, x_value, y_value):
        self.xlist.append(x_value)
        self.ylist.append(y_value)
        plt.plot(self.xlist, self.ylist, self.style, color=self.color)
        plt.pause(0.001)
        

if __name__ == '__main__':
    N=100
    monitor = dynamic_pic("Dynamic representation", xlabel="x", ylabel="y")
    for i in range(N):
        # 放绘图代码 draw
        monitor(i*0.1, np.sin(i*0.1))
    plt.show()