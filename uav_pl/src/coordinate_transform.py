from math import atan2, tan, sqrt
from geopy.distance import distance



def transform_to_ground_xy(error_img_xy, altitude ,fov_hv):

    """Transform the error from the image plane to the ground plane using the altitude and the angle of the UAV. 
    The error is given in pixels. The angle of the UAV is given in radians. The altitude is given in meters. 
    The output is the error in the ground plane in meters."""
    error_ground_xy = [0, 0] # Error in the ground plane
    for idx in range(2):
        img_angle = fov_hv[idx]/2 * error_img_xy[idx] # Angle of the error respective to camera 
        error_ground_xy[idx] = altitude * tan(img_angle) # Error in the ground plane
    return error_ground_xy

def transform_ground_to_img_xy(error_ground_xy, altitude, fov_hv, img_size):
    """Transform the error from the ground plane to the image plane using the altitude and the angle of the UAV. 
    The error is given in meters. The altitude is given in meters. 
    The output is the error in the image plane in pixels. The angle of the UAV is negleted. This is only for debugging purposes."""

    angles = [0, 0] # Angle of the error respective to camera
    error_px_xy = [0, 0] # Error in the image plane
    for idx in range(2):
        angles[idx] = atan2(error_ground_xy[idx], altitude) # Angle of the error respective to camera
        error_px_xy[idx] = int((angles[idx]/(fov_hv[idx])+0.5) * img_size[idx]) # Error in the image plane
    return error_px_xy


def calculate_new_coordinate(current_lat, current_lon, error_ground_xy, heading):
    """Calculate the new coordinate using the error and the current coordinate"""
    distance_km = sqrt(error_ground_xy[0]**2 + error_ground_xy[1]**2) / 1000 #Calculate distance in km
    heading_in_image = atan2(error_ground_xy[0], error_ground_xy[1]) #Calculate angle in image relative to y axis
    bearing = (heading + heading_in_image)*180/3.14159 #Calculate bearing in degrees
    destination = distance(kilometers=distance_km).destination((current_lat,current_lon), bearing) #Calculate the destination using geopy
    return destination.latitude, destination.longitude

def calculate_size_in_px(altitude, size_object_m, cam_hfov, image_width):
    """Calculate the size of the object in pixels"""
    size_img_on_ground = tan(cam_hfov/2)*altitude #This is the image width on the ground in meters
    rel_size = size_object_m/size_img_on_ground #This is the size of the bigger circle compared to the overall frame
    size_obj = rel_size*image_width #This is the radius of the bigger circle in pixels
    return size_obj

def calculate_altitude(length_px, cam_hfov, img_width, actual_length):
    """Calculate the altitude of the UAV using the length of the object in pixels and the actual length of the object in meters."""
    angle_per_px = cam_hfov/img_width #This is the angle per pixel in radians
    angle_object = length_px*angle_per_px
    altitude = (actual_length)/(2*tan(angle_object/2)) 
    return altitude



    