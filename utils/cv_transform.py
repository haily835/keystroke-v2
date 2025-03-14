import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the image
image = cv2.imread('datasets/shortvideo-1/raw_frames/video_0/frame_12.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# List to store the clicked points
# Sequence of the points will be transform (top left, top right, bottom left, bottom right)
clicked_points = []

# Function to handle mouse clicks
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        # Append the point as a (x, y) tuple
        clicked_points.append((event.xdata, event.ydata))
        # Draw a red dot at the clicked position
        plt.scatter(event.xdata, event.ydata, color='red')
        plt.draw()

        # Once 4 points are selected, apply the perspective transform
        if len(clicked_points) == 4:
            plt.close()  # Close the figure
            perform_perspective_transform(clicked_points)

# Function to perform perspective transform using the selected points
def perform_perspective_transform(points):
    src_points = np.float32(points)
    dst_points = np.float32([[0, 0], [600, 0], [0, 300], [600, 300]])

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transformation
    transformed_image = cv2.warpPerspective(image, matrix, (600, 300))

    # Display the transformed image
    plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
    plt.title('Transformed Image')
    plt.axis('off')
    plt.show()

# Display the image and set up the click event
fig, ax = plt.subplots()
ax.imshow(image_rgb)
ax.set_title('Click to select 4 points')
ax.axis('off')

# Connect the click event to the onclick function
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()