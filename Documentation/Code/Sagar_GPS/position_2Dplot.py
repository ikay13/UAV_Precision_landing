#!/bin/env python3

import argparse
import json
from matplotlib import pyplot as plt

class Position2DPlots:
    def __init__(self, x_file_path, y_file_path, z_file_path, time_file_path):
        self.label = ['GPS local','Odometry', 'Pose', 'Lidar']
        self.line = [[], []]

        with open(x_file_path, 'r') as file:
            self.x_position = json.loads(file.read())
        with open(y_file_path, 'r') as file:
            self.y_position = json.loads(file.read())
        with open(z_file_path, 'r') as file:
            self.z_position = json.loads(file.read())
        with open(time_file_path, 'r') as file:
            self.time = json.loads(file.read())

        self.fig2 = plt.figure()


        self.ax2 = self.fig2.add_subplot(1, 1, 1)
        self.setup_2d_plot(self.ax2, self.time, self.z_position, 't in s', 'Altitude in m')

        plt.show()

    def setup_2d_plot(self, ax, xvalues, yvalues, xlabel, ylabel):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        #Plot only specific time range
        time_start = 60 #t in s
        time_end = 100 #t in s
        start_index = 0
        end_index = 0
        

        for i in range(len(self.label)):
            for i in range(len(self.time)):
                if self.time[i] >= time_start:
                    start_index = i
                    break
            for i in range(len(self.time)):
                if self.time[i] >= time_end:
                    end_index = i
                    break
            print("label", self.label[i])
            if not self.label[i] == "Pose" and not self.label[i] == "Odometry":
                ax.plot(xvalues[start_index:end_index], yvalues[self.label[i]][start_index:end_index], label=self.label[i])
                #ax.plot(xvalues[self.label[i]], yvalues[self.label[i]], label=self.label[i])

        ax.legend()
        # ax.axis('equal')

if __name__ == '__main__':
    path = 'Documentation/Code/Sagar_GPS/'

    x_file_path = path + 'x_pos.txt'
    y_file_path = path + 'y_pos.txt'
    z_file_path = path + 'z_pos.txt'
    time_file_path = path + 'time.txt'

    Position2DPlots(x_file_path, y_file_path, z_file_path, time_file_path)
