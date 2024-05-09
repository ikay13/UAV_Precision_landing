#!/usr/bin/env python3

import argparse
import os

parser = argparse.ArgumentParser(description='images to video')
parser.add_argument('bag', type=str, help='Bag name without .bag')
parser.add_argument('frame_rate', type=int, help='output video frame rate')
args = parser.parse_args()
 
path = '~/ROS/src/UAV_Precision_landing/uav_pl/Videos'
command = "ffmpeg -framerate {frame_rate} -i {path}/frame%04d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {bag_name}.mp4".format(frame_rate=args.frame_rate, path=path, bag_name=args.bag)

os.system("mv ~/.ros/frame*.jpg {}".format(path))

if os.getcwd()==path:
	os.system(command)
else:
	os.system('cd {} && mkdir {} && {} && mv frame*.jpg {}'.format(path, args.bag, command, args.bag))
