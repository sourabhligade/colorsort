import cv2
import numpy as np
from collections import Counter
import webcolors

# Function to get the nearest color name for an RGB value
def closest_color(rgb):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - rgb[0]) ** 2
        gd = (g_c - rgb[1]) ** 2
        bd = (b_c - rgb[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

# Load the image in BGR format
image = cv2.imread('/Users/sourabhligade/Documents/imagecolor/car3.png')
if image is None:
    print("Error: Unable to load image.")
    exit()

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape image into a 2D array of pixels
pixels = image_rgb.reshape((-1, 3))

# Convert pixels to float32 for k-means clustering
pixels = np.float32(pixels)

# Define criteria and apply k-means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Experiment with different values of k
for k in range(3, 11):
    _, labels, centroids = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centroids to integer
    centroids = np.uint8(centroids)

    # Count the occurrences of each label
    label_counts = Counter(labels.flatten())

    # Find the label with the highest count
    most_common_label = max(label_counts, key=label_counts.get)

    # Get the centroid corresponding to the most common label
    dominant_color = centroids[most_common_label]

    # Convert the centroid to a tuple for webcolors
    dominant_color_tuple = tuple(dominant_color.tolist())

    # Get the closest color name for the dominant color
    color_name = closest_color(dominant_color_tuple)

    # Print the dominant color and its name
    print(f"Dominant color (k={k}): {dominant_color} - Name: {'red' if color_name == 'crimson' else color_name}")
