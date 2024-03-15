import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

def ecg_gridest_matchedfilt(img, params=None):
    """
    Estimates grid size in ECG images using matched filtering.

    Parameters:
    img (numpy.ndarray): A 2D matrix representing the ECG image in grayscale.
    params (dict): A dictionary containing various parameters to control
                   the image processing and grid detection algorithm. Default
                   values are used if this argument is not provided. See below for details.

    Returns:
    tuple: Tuple containing grid sizes, grid size prominences, mask sizes,
           average matched filter output powers, and indices of selected local peaks.

    """

    # parse algorithm parameters
    if params is None:
        params = {}

    blur_sigma_in_inch = params.get('blur_sigma_in_inch', 1.0)
    paper_size_in_inch = params.get('paper_size_in_inch', [11, 8.5])
    remove_shadows = params.get('remove_shadows', True)
    sat_pre_grid_det = params.get('sat_pre_grid_det', True)
    sat_level_pre_grid_det = params.get('sat_level_pre_grid_det', 0.7)
    num_seg_hor = params.get('num_seg_hor', 4)
    num_seg_ver = params.get('num_seg_ver', 4)
    tiling_method = params.get('tiling_method', 'RANDOM_TILING')
    total_segments = params.get('total_segments', 16)
    max_grid_size = params.get('max_grid_size', 30)
    min_grid_size = params.get('min_grid_size', 2)
    power_avg_prctile_th = params.get('power_avg_prctile_th', 95.0)
    detailed_plots = params.get('detailed_plots', 0)

    width, height = img.shape[1], img.shape[0]

    # convert image to gray scale
    img_gray = np.mean(img, axis=2) if img.ndim == 3 else img.astype(np.float64)
    img_gray = img_gray / np.max(img_gray)

    # shaddow removal and intensity normalization
    if remove_shadows:
        blur_sigma = np.mean([width * blur_sigma_in_inch / paper_size_in_inch[0],
                              height * blur_sigma_in_inch / paper_size_in_inch[1]])
        img_gray_blurred = gaussian_filter(img_gray, blur_sigma, mode='constant')
        img_gray_normalized = img_gray / img_gray_blurred
        img_gray_normalized = (img_gray_normalized - np.min(img_gray_normalized)) / \
                              (np.max(img_gray_normalized) - np.min(img_gray_normalized))
    else:
        img_gray_blurred = img_gray
        img_gray_normalized = img_gray

    # image density saturation
    if sat_pre_grid_det:
        img_sat = np.tanh(1.0 - img_gray_normalized)
        img_gray_normalized = img_sat

    # segmentation
    seg_width = width // num_seg_hor
    seg_height = height // num_seg_ver
    mask_size = np.arange(min_grid_size, max_grid_size + 1)
    matched_filter_powers = np.zeros((num_seg_hor * num_seg_ver, len(mask_size)))

    # define boundary mask
    def boundary_mask(sz):
        B = np.zeros((sz, sz))
        B[0, :] = 1
        B[-1, :] = 1
        B[:, 0] = 1
        B[:, -1] = 1
        return B / np.sum(B)

    for i in range(num_seg_ver):
        for j in range(num_seg_hor):
            segment = img_gray_normalized[i * seg_height:(i + 1) * seg_height,
                      j * seg_width:(j + 1) * seg_width]
            segment = (segment - np.mean(segment)) / np.std(segment)
            for g, mask_sz in enumerate(mask_size):
                B = boundary_mask(mask_sz)
                B = B - np.mean(B)
                matched_filtered = np.abs(np.convolve(segment.ravel(), B.ravel(), mode='same')).reshape(segment.shape)
                pm = matched_filtered.ravel() ** 2
                pm_th = np.percentile(pm, power_avg_prctile_th)
                matched_filter_powers[i * num_seg_hor + j, g] = 10 * np.log10(np.mean(pm[pm > pm_th]))

    matched_filter_powers_avg = np.mean(matched_filter_powers, axis=0)
    peaks, _ = find_peaks(matched_filter_powers_avg)
    grid_sizes = mask_size[peaks] - 1  # -1 is to convert mask size to period
    grid_size_prominences = np.array([matched_filter_powers_avg[p] for p in peaks])

    # Plot results
    if detailed_plots > 0:
        # Plot matched filter output powers
        import matplotlib.pyplot as plt

        plt.figure()
        for i in range(num_seg_hor * num_seg_ver):
            plt.plot(mask_size - 1, matched_filter_powers[i, :])
        plt.plot(mask_size - 1, matched_filter_powers_avg, 'k', linewidth=3)
        plt.plot(mask_size[peaks] - 1, matched_filter_powers_avg[peaks], 'ro', markersize=18)
        plt.grid()
        plt.xlabel('Grid size')
        plt.ylabel('Average power (dB)')
        plt.title('Average matched-filter output power vs grid size')

        # Plot preprocessing stages
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('img')
        plt.subplot(2, 2, 2)
        plt.imshow(img_gray, cmap='gray')
        plt.title('img_gray')
        plt.subplot(2, 2, 3)
        plt.imshow(img_gray_blurred, cmap='gray')
        plt.title('img_gray_blurred')
        plt.subplot(2, 2, 4)
        plt.imshow(img_gray_normalized, cmap='gray')
        plt.title('img_gray_normalized')
        plt.suptitle('Preprocessing stages (shaddow removal and intensity normalization)')
        plt.show()

    return grid_sizes, grid_size_prominences, mask_size, matched_filter_powers_avg, peaks
