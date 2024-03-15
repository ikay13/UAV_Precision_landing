from math import atan2, tan, sqrt
from geopy.distance import distance
def transform_to_ground_xy(error_img_xy, angle_uav_xy, altitude):
    """Transform the error from the image plane to the ground plane using the altitude and the angle of the UAV. 
    The error is given in pixels. The angle of the UAV is given in radians. The altitude is given in meters. 
    The output is the error in the ground plane in meters."""
    d = 1.55 # Distance from camera to virtual image plane
    error_ground_xy = [0, 0] # Error in the ground plane
    for idx in range(2):
        img_angle = atan2(error_img_xy[idx], d) # Angle of the error respective to camera
        error_ground_xy[idx] = altitude * tan(img_angle + angle_uav_xy[idx]) # Error in the ground plane
    return error_ground_xy

def calculate_new_coordinate(current_lat, current_lon, error_ground_xy, heading):
    """Calculate the new coordinate using the error and the current coordinate"""
    distance_km = sqrt(error_ground_xy[0]**2 + error_ground_xy[1]**2) / 1000 #Calculate distance in km
    heading_in_image = atan2(error_ground_xy[0], error_ground_xy[1]) #Calculate angle in image relative to y axis
    bearing = (heading + heading_in_image)*180/3.14159 #Calculate bearing in degrees
    destination = distance(kilometers=distance_km).destination((current_lat,current_lon), bearing) #Calculate the destination using geopy
    return destination.latitude, destination.longitude

def calculate_size_in_px(altitude, size_object_m, cam_hfov, image_width):
    """Calculate the size of the object in pixels"""
    size_img_on_ground = tan(cam_hfov/2)*2*altitude #This is the image width on the ground in meters
    rel_size = size_object_m/size_img_on_ground #This is the size of the bigger circle compared to the overall frame
    size_obj = rel_size*image_width #This is the radius of the bigger circle in pixels
    return size_obj

    

    