import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from image_to_sequence import image_to_sequence  # Assuming you have a Python function named image_to_sequence

# Define the path to the folder containing ECG image segments
data_path = '../../sample-data/ecg-images/sample-segments/'

# Get a list of all files in the image folder
all_files = os.listdir(data_path)

# Loop over all files, reading and processing each image
for file_name in all_files:
    image_fname = os.path.join(data_path, file_name)

    try:
        # Read the image
        img = cv2.imread(image_fname)

        # Apply different sequence extraction methods to the image
        z0 = image_to_sequence(img, 'dark-foreground', 'max_finder', [], False)
        z1 = image_to_sequence(img, 'dark-foreground', 'hor_smoothing', 3)
        z2 = image_to_sequence(img, 'dark-foreground', 'all_left_right_neighbors')
        z3 = image_to_sequence(img, 'dark-foreground', 'combined_all_neighbors')
        z4 = image_to_sequence(img, 'dark-foreground', 'moving_average', 3)

        # Combine results from all methods
        z_combined = np.median(np.concatenate((z0, z1, z2, z3, z4), axis=0), axis=0)

        # Prepare for plotting
        nn = np.arange(img.shape[1])
        img_height = img.shape[0]

        # Display the original image and overlay the extracted sequences
        plt.figure()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.plot(nn, img_height - z0, linewidth=3, label='max_finder')
        plt.plot(nn, img_height - z1, linewidth=3, label='hor_smoothing')
        plt.plot(nn, img_height - z2, linewidth=3, label='all_left_right_neighbors')
        plt.plot(nn, img_height - z3, linewidth=3, label='combined_all_neighbors')
        plt.plot(nn, img_height - z4, linewidth=3, label='moving_average')
        plt.plot(nn, img_height - z_combined, linewidth=3, label='combined methods')

        # Add legend and title
        plt.legend()
        plt.title(f'Paper ECG vs recovered signal for: {file_name}')
        plt.xlabel('Time')
        plt.ylabel('Intensity')
        plt.show()

        # Uncomment the following line to save the superposed image, if needed
        # plt.savefig(fname[:-4] + '-rec.png')
        plt.close()

    except:
        print(f'Warning: File {file_name} not an image, or processing failed')
