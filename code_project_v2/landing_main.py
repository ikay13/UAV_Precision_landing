from square_detect import detect_square_main
from coordinate_transform import transform_to_ground_xy, calculate_new_coordinate

import cv2 as cv
import numpy as np


class error_estimation_image:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.timeout = 0

class uav:
    def __init__(self):
        self.angle_x = 0
        self.angle_y = 0
        self.altitude = 10
        self.heading = 0
        self.latitude = 29.183972
        self.longitude = -81.043251

def main():
    video = cv.VideoCapture("code_project_v2/images/cut.mp4")
    err_estimation = error_estimation_image()  # Create an instance of the error_estimation class
    uav_inst = uav()  # Create an instance of the uav_angles class
    while True:
        ret, frame = video.read()
        if ret:
            frame = cv.resize(frame, (640, 480))
            error_xy = detect_square_main(frame, 10)
            if error_xy is not None:
                err_estimation.x = error_xy[0]
                err_estimation.y = error_xy[1]
                error_ground_xy = transform_to_ground_xy([err_estimation.x, err_estimation.y], [uav_inst.angle_x, uav_inst.angle_y], uav_inst.altitude)
                target_lat, target_lon = calculate_new_coordinate(uav_inst.latitude, uav_inst.longitude, error_ground_xy, uav_inst.heading)
        else:
            break

      

if __name__ == "__main__":
    main()
