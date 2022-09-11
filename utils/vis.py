import os
import pdb
import time
import torch
import random 

import numpy as np
import matplotlib.pyplot as plt


class dynamic_pic():
    def __init__(self , capacity=5000):
        self.capacity = capacity
        self.data_base = dict()
        self.cnames = ["k", "r", "gold", "g", "dodgerblue", "slateblue"]

    def draw(self, joint:list=None, row_max=3, pause=1, save_path=None):
        """drop_x for the no x data"""
        plt.close()
        if joint is None:
            joint = list(self.data_base.keys())
        sub_num = len(joint)
        row = np.ceil(sub_num / row_max).astype(int)

        plt.figure(figsize=(row_max * 9, row * 6))
        num = 1
        
        for item in joint:
            try:
                if isinstance(item, str):
                    plt.subplot(row, row_max, num)
                    if self.data_base[item]["mode"] == "line":
                        plt.plot(self.data_base[item]["x"], self.data_base[item]["y"], label=item)
                    elif self.data_base[item]["mode"] == "scatter":
                        plt.scatter(self.data_base[item]["x"], self.data_base[item]["y"], label=item, marker=".", linewidths=0.3)
                    elif self.data_base[item]["mode"] == "figure":
                        plt.imshow(self.data_base[item]["x"])
                        plt.axis(False)
                elif isinstance(item, list):
                    plt.subplot(row, row_max, num)
                    for i, it in enumerate(item):
                        if self.data_base[it]["mode"] == "line":
                            plt.plot(self.data_base[it]["x"], self.data_base[it]["y"], label=it, color=self.cnames[i])
                        else:
                            plt.scatter(self.data_base[it]["x"], self.data_base[it]["y"], label=it, marker=".", color=self.cnames[i], linewidths=0.3)
                plt.legend(loc="best")
                num += 1
            except KeyError:
                print("\r Key {} not found".format(item), end="", flush=True)
                num += 1

        plt.tight_layout()
        if pause != 0:
            plt.pause(pause)
        elif save_path is not None:
            plt.savefig(save_path)
        
    def _write(self, x,  y, category:str="base", mode="line", drop_mode="drop", drop_x=False):
        """mode only received as 'line' of 'scatter' , drop mode: drop or jump"""
        if category not in self.data_base:
            self.data_base[category] = {"x":[], "y":[], "mode":mode}
        
        if mode == "figure":
            self.data_base[category]["x"] = x
            self.data_base[category]["y"] = 0
        else:
            if len(self.data_base[category]["x"]) >= self.capacity:
                if mode == "line":
                    if drop_mode == "jump":
                        self.data_base[category]["x"] = [self.data_base[category]["x"][i] for i in range(self.capacity) if i % 2 == 0]
                        self.data_base[category]["y"] = [self.data_base[category]["y"][i] for i in range(self.capacity) if i % 2 == 0]
                    elif drop_mode == "drop":
                        del self.data_base[category]["x"][0:int(self.capacity / 2)]
                        del self.data_base[category]["y"][0:int(self.capacity / 2)]
                elif mode == "scatter":
                    del self.data_base[category]["x"][0:int(self.capacity / 2)]
                    del self.data_base[category]["y"][0:int(self.capacity / 2)]
                else:
                    raise ValueError("mode only received as base line or scatter")

            if isinstance(y, list):
                self.data_base[category]["x"] += x
                self.data_base[category]["y"] += y
            else:
                self.data_base[category]["x"].append(x)
                self.data_base[category]["y"].append(y)

            if drop_x:
                self.data_base[category]["x"] = [i for i in range(len(self.data_base[category]["y"]))]

    def clean(self, key="all"):
        if isinstance(key, str):
            if key == "all":
                self.data_base = dict()
            else:
                if key not in self.data_base:
                    return
                self.data_base[key]["x"] = []
                self.data_base[key]["y"] = []
        elif isinstance(key, list):
            for item in key:
                self.clean(item)

    def __call__(self, x, y, category:str="base", mode="line", drop_mode="drop", drop_x=False):
        self._write(x, y, category, mode, drop_mode, drop_x)

if __name__ == '__main__':
    p1 = dynamic_pic(capacity=3000)
    for i in range(1000):
        # print((torch.randn(3, 2) / 5 + 2).numpy().T)
        x1, y1 = (torch.randn(3, 2) / 5 + 2).numpy().T
        x, y = (torch.randn(3, 2) / 5).numpy().T
        p1(x, y, category="a", mode="scatter")
        p1(x1, y1, category="b", mode="scatter")
        # if i % 10 == 0:
        p1.draw(joint=[["a", "b"]], row_max=1)