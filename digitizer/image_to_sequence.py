import numpy as np
import cv2

def image_to_sequence(img, mode, method, windowlen=3, plot_result=False):
    """
    Extracts a sequence/time-series from an image.

    This function processes an image to extract a time-series representation,
    for example an ECG images. The method to extract the sequence depends on
    the image's characteristics (e.g., whether the foreground is darker or
    brighter than the background) and the filtering approach. The function
    returns a vector that has the same length as the width of the input image
    (the second dimension of the input image matrix). The method used
    for extracting the sequence can be justified using a maximum likelihood
    estimate of adjacent temporal samples when studied in a probabilistic
    framework.

    Args:
    - img: A 2D matrix representing the image.
    - mode: A string specifying the foreground type: 'dark-foreground' or 'bright-foreground'.
    - method: A string specifying the filtering method to use. Options are
              'max_finder', 'moving_average', 'hor_smoothing',
              'all_left_right_neighbors', 'combined_all_neighbors'.
    - windowlen: (optional) Length of the moving average window. Default is 3.
    - plot_result: (optional) Boolean to plot the result. Default is False.

    Returns:
    - data: Extracted sequence or time-series from the image.
    """

    # Convert image to grayscale if it's in RGB format
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # Process image based on specified foreground mode
    if mode == 'dark-foreground':
        img_flipped = cv2.bitwise_not(img_gray)
    elif mode == 'bright-foreground':
        img_flipped = img_gray
    else:
        raise ValueError("Invalid mode. Mode must be 'dark-foreground' or 'bright-foreground'.")

    # Apply different methods for sequence extraction
    if method == 'max_finder':
        img_filtered = img_flipped
    elif method == 'moving_average':
        h = np.ones((windowlen,)) / windowlen
        img_filtered = cv2.filter2D(img_flipped, -1, h)
    elif method == 'hor_smoothing':
        h = np.ones((1, windowlen)) / windowlen
        img_filtered = cv2.filter2D(img_flipped, -1, h)
    elif method == 'all_left_right_neighbors':
        h = np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]]) / 5
        img_filtered = cv2.filter2D(img_flipped, -1, h)
    elif method == 'combined_all_neighbors':
        h1 = np.array([[1, 1, 1]])
        h2 = np.array([[1], [1], [1]])
        h3 = np.eye(3, dtype=np.uint8)
        z1 = cv2.filter2D(img_flipped, -1, h1)
        z2 = cv2.filter2D(img_flipped, -1, h2)
        z3 = cv2.filter2D(img_flipped, -1, h3)
        img_filtered = np.minimum(np.minimum(z1, z2), z3)
    else:
        raise ValueError("Invalid method.")

    # Find the maximum pixel value in each column to represent the ECG signal
    _, I = cv2.minMaxLoc(img_filtered, None)
    img_height = img_filtered.shape[0]
    data = img_height - I[1]  # Convert to vertical position (ECG amplitude with offset)

    # Plot the result if requested
    if plot_result:
        cv2.line(img, (0, data), (img.shape[1], data), (0, 255, 0), thickness=3)
        cv2.imshow('Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return data
