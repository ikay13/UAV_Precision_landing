from math import atan2, pi
import cv2 as cv
def display_error_and_text(frame, curr_err_px_tuple, alt_from_img, uav_inst):
    """Display the error in pixels and the altitude on the frame"""
    ret_rame = frame.copy()
    curr_err_px = list(curr_err_px_tuple)
    altitude = uav_inst.altitude
    state = uav_inst.state
    list_states = ["initial",
        "fly_to_waypoint",
        "detect_square",
        "fly_to_target",
        "check_for_target",
        "descend_square",
        "descend_concentric",
        "descend_inner_circle",
        "detect_tins",
        "fly_over_tins",
        "land_tins",
        "return_to_launch",
        "climb_to_altitude",
        "done",
        "align_before_landing",
        "align_tin"]
    curr_err_px[1] = frame.shape[0]-curr_err_px[1]  # Invert the y axis
    # Draw arrow from the middle to curr_err_px
    arrow_color = (0, 0, 255)  # Red color
    text_color = (0, 255, 255)  # Red color
    arrow_thickness = 2
    arrow_tip_length = 0.15

    # Calculate the middle point of the frame
    frame_height, frame_width = frame.shape[:2]
    middle_point = (frame_width // 2 - frame_width //4, frame_height // 2)

    # Calculate the end point of the arrow
    end_point = (curr_err_px[0], curr_err_px[1])

    texts = [f"State: {list_states[state]}", f"Altitude from FC: {altitude:.2f}", f"Altitude from image: {alt_from_img:.2f}"]
    # # Calculate the angle of the arrow
    # angle = atan2(curr_err_px[1], curr_err_px[0]) * 180 / pi

    # Draw the arrow on the frame
    if curr_err_px != [0, frame.shape[0]]: #Default value meaning object not detected
        # print("curr_err", curr_err_px)
        cv.arrowedLine(ret_rame, middle_point, end_point, arrow_color, arrow_thickness, cv.LINE_AA, tipLength=arrow_tip_length)
    else:
        texts.append("Not detected")

    for idx, text in enumerate(texts):
        cv.putText(ret_rame, text, (10, 20+idx*20), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv.LINE_AA)
    return ret_rame
        