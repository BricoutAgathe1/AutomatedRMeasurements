import cv2
import math

# Global variables to store selected points
points = []
selected_points = []

def mouse_callback(event, x, y, flags, param):
    global points, selected_points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("image", image)

        # Check if two points have been selected
        if len(points) == 2:
            selected_points = points.copy()
            points = []  # Reset points for next selection

            # Calculate distance
            distance = math.sqrt((selected_points[1][0] - selected_points[0][0])**2 +
                                 (selected_points[1][1] - selected_points[0][1])**2)
            print(f"Distance between points: {distance} pixels")


# Load an image
image_path = 'dataset/images/frame0.jpg'
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# Clone the image for display
clone = image.copy()

# Create a window and set mouse callback
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_callback)

# Display the image
cv2.imshow("image", image)

# Wait for ESC key to exit
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break

# Close all windows
cv2.destroyAllWindows()
