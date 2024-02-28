from math import atan2, tan, sqrt
from geopy.distance import distance
def transform_to_ground_xy(error_img_xy, angle_uav_xy, altitude):
    d = 1.55 # Distance from camera to virtual image plane
    error_ground_xy = [0, 0] # Error in the ground plane
    for idx in range(2):
        img_angle = atan2(error_img_xy[idx], d) # Angle of the error respective to camera
        error_ground_xy[idx] = altitude * tan(img_angle + angle_uav_xy[idx]) # Error in the ground plane
    return error_ground_xy

def calculate_new_coordinate(current_lat, current_lon, error_ground_xy, heading):
    distance_km = sqrt(error_ground_xy[0]**2 + error_ground_xy[1]**2) / 1000 #Calculate distance in km
    heading_in_image = atan2(error_ground_xy[0], error_ground_xy[1]) #Calculate angle in image relative to y axis
    bearing = (heading + heading_in_image)*180/3.14159 #Calculate bearing in degrees
    destination = distance(kilometers=distance_km).destination((current_lat,current_lon), bearing) #Calculate the destination using geopy
    

    return destination.latitude, destination.longitude

    