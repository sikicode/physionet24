import os
import cv2
import numpy as np

from ecg_gridest_margdist import ecg_gridest_margdist  # Assuming you have a Python function for this
from ecg_gridest_spectral import ecg_gridest_spectral  # Assuming you have a Python function for this
from ecg_gridest_matchedfilt import ecg_gridest_matchedfilt  # Assuming you have a Python function for this

# Define the path to the folder containing ECG images
data_path = '../../sample-data/ecg-images/'

# Get a list of all files in the image folder
all_files = os.listdir(data_path)

# Loop over all files, reading them in
for file_name in all_files:
    image_fname = os.path.join(data_path, file_name)
    try:
        img = cv2.imread(image_fname)

        # Estimate grid resolution based on paper-size
        paper_size = [11.0, 8.5]
        coarse_grid_size_paper_based, fine_grid_size_paper_based = ecg_grid_size_from_paper(img, paper_size[0], 'in')

        # Marginal distribution-based method
        # Setting all the parameters for ecg_gridest_margdist function
        params_margdist = {
            'blur_sigma_in_inch': 1.0,
            'paper_size_in_inch': paper_size,
            'remove_shadows': True,
            'apply_edge_detection': False,
            'post_edge_det_gauss_filt_std': 0.01,
            'post_edge_det_sat': True,
            'sat_level_upper_prctile': 99.0,
            'sat_level_lower_prctile': 1.0,
            'sat_pre_grid_det': False,
            'sat_level_pre_grid_det': 0.7,
            'num_seg_hor': 4,
            'num_seg_ver': 4,
            'hist_grid_det_method': 'RANDOM_TILING',
            'total_segments': 100,
            'min_grid_resolution': 1,
            'min_grid_peak_prom_prctile': 2.0,
            'cluster_peaks': True,
            'max_clusters': 3,
            'cluster_selection_method': 'GAP_MIN_VAR',
            'avg_quartile': 50.0,
            'detailed_plots': 1
        }
        gridsize_hor_margdist, gridsize_ver_margdist, grid_spacings_hor, grid_spacing_ver = ecg_gridest_margdist(img, params_margdist)

        # Spectral-based method
        # Setting all the parameters for ecg_gridest_spectral function
        params_spectral = {
            'blur_sigma_in_inch': 1.0,
            'paper_size_in_inch': paper_size,
            'remove_shadows': True,
            'apply_edge_detection': False,
            'post_edge_det_gauss_filt_std': 0.01,
            'post_edge_det_sat': False,
            'sat_level_upper_prctile': 99.0,
            'sat_level_lower_prctile': 1.0,
            'sat_pre_grid_det': False,
            'sat_level_pre_grid_det': 0.7,
            'num_seg_hor': 4,
            'num_seg_ver': 4,
            'spectral_tiling_method': 'RANDOM_TILING',
            'total_segments': 100,
            'min_grid_resolution': 1,
            'min_grid_peak_prominence': 1.0,
            'detailed_plots': 1
        }
        gridsize_hor_spectral, gridsize_ver_spectral = ecg_gridest_spectral(img, params_spectral)

        # Matched filter-based method
        params_matchfilt = params_margdist.copy()
        params_matchfilt['sat_pre_grid_det'] = True
        params_matchfilt['sat_level_pre_grid_det'] = 0.7
        params_matchfilt['total_segments'] = 10
        params_matchfilt['tiling_method'] = 'RANDOM_TILING'
        grid_sizes_matchedfilt, grid_size_prom_matchedfilt, mask_size_matchedfilt, matchedfilt_powers_avg, I_peaks_matchedfilt = ecg_gridest_matchedfilt(img, params_matchfilt)

        print(f'Grid resolution estimate per 0.1mV x 40ms (paper size-based): {fine_grid_size_paper_based} pixels')
        print(f'Grid resolution estimates per 0.1mV x 40ms (matched filter-based): {grid_sizes_matchedfilt} pixels')
        print(f'Horizontal grid resolution estimate (margdist): {gridsize_hor_margdist} pixels')
        print(f'Vertical grid resolution estimate (margdist): {gridsize_ver_margdist} pixels')
        print(f'Horizontal grid resolution estimate (spectral): {gridsize_hor_spectral} pixels')
        print(f'Vertical grid resolution estimate (spectral): {gridsize_ver_spectral} pixels')
        closest_ind_hor = np.argmin(np.abs(np.array(gridsize_hor_spectral) - fine_grid_size_paper_based))
        closest_ind_ver = np.argmin(np.abs(np.array(gridsize_ver_spectral) - fine_grid_size_paper_based))
        print(f'Closest spectral horizontal grid resolution estimate from paper-based resolution (per 0.1mV x 40ms): {gridsize_hor_spectral[closest_ind_hor]} pixels')
        print(f'Closest spectral vertical grid resolution estimate from paper-based resolution (per 0.1mV x 40ms): {gridsize_ver_spectral[closest_ind_ver]} pixels')

        print('---')

        # Close all figures
        plt.close('all')

    except Exception as e:
        print(f'Error processing file {image_fname}: {e}')
