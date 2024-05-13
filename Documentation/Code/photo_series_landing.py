import cv2
import numpy as np

# Load images
folder_path = "/home/mathis_ros/Pictures/Photo_serie_landing_sim/"
image_paths = [f"{folder_path}{i}.png" for i in range(1, 11)]
images = [cv2.imread(path) for path in image_paths]
#Assert not empty
for img in images:
    if img is None:
        print("Error: Image could not be read.")
        exit()
cv2.imshow('Path Taken', images[0])

# Create a blank image that will hold the final result
height, width, channels = images[0].shape
final_image = np.zeros((height, width, channels), dtype=np.uint8)

# Add each image to the final image
for img in images:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)  # Adjust threshold as needed
    final_image = cv2.add(final_image, cv2.bitwise_and(img, img, mask=mask))

# Save or display the final image
cv2.imwrite('final_path.jpg', final_image)
cv2.imshow('Path Taken', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
