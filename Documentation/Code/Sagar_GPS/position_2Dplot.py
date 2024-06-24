#!/bin/env python3

import argparse
import json
from matplotlib import pyplot as plt

class Position2DPlots:
    def __init__(self, x_file_path, y_file_path, z_file_path, time_file_path):
        self.label = ['GPS local', 'Lidar']
        
        # Read data from files
        with open(x_file_path, 'r') as file:
            self.x_position = json.loads(file.read())
        with open(y_file_path, 'r') as file:
            self.y_position = json.loads(file.read())
        with open(z_file_path, 'r') as file:
            self.z_position = json.loads(file.read())
        with open(time_file_path, 'r') as file:
            self.time_data = json.loads(file.read())
            self.time = self.time_data['GPS local']  # Extract the list of time values

        # Check if all files have been opened and contain data
        if not self.x_position or not self.y_position or not self.z_position or not self.time:
            print("Error opening files or files are empty")
            return
        
        # Setup plot
        self.fig2 = plt.figure()
        self.ax2 = self.fig2.add_subplot(1, 1, 1)
        self.setup_2d_plot(self.ax2, self.time, self.z_position, 't in s', 'Altitude in m')

        

        # Save and show the plot
        #Set size of figure
        self.fig2.set_size_inches(10, 6)
        output_file_path = 'Documentation/Images/finished/altitude_vs_time_gps_lidar.png'
        plt.savefig(output_file_path, bbox_inches='tight')
        print(f"Plot saved as {output_file_path}")

        plt.show()


    def setup_2d_plot(self, ax, xvalues, yvalues, xlabel, ylabel):
        # Set x and y labels with latex
        ax.set_xlabel(r'\textit{t} in s')
        ax.set_ylabel(r'Altitude in m')
        # ax.set_xlabel(xlabel)
        # ax.set_ylabel(ylabel)
        ax.grid(True)

        # Plot only specific time range
        time_start = 52  # t in s
        time_end = 120 # t in s
        start_index = next(i for i, t in enumerate(xvalues) if t >= time_start)
        end_index = next(i for i, t in enumerate(xvalues) if t >= time_end)

        for label in self.label:
            print("label", label)
            if label in yvalues:
                if label == 'GPS local':
                    color_line = 'k'
                else:
                    #grey
                    color_line = '#606060'
                ax.plot(xvalues[start_index:end_index], yvalues[label][start_index:end_index], label=label, color=color_line)
        
        #set start and end of plot
        ax.set_xlim([time_start, time_end])
        ax.legend()

if __name__ == '__main__':
    # Set common style properties
    plt.rc('text', usetex=True)
    plt.rcParams.update({
        'font.size': 26,
        'axes.labelsize': 26,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 22,
        'lines.linewidth': 2.5,
        'grid.linewidth': 1.5,
    })

    path = 'Documentation/Code/Sagar_GPS/'

    x_file_path = path + 'x_pos.txt'
    y_file_path = path + 'y_pos.txt'
    z_file_path = path + 'z_pos.txt'
    time_file_path = path + 'time.txt'

    Position2DPlots(x_file_path, y_file_path, z_file_path, time_file_path)
