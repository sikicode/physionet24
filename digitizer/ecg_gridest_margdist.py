import numpy as np
import cv2
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from scipy.stats import trim_mean

def ecg_gridest_margdist(img, params=None):
    # Default parameters
    if params is None:
        params = {}

    if 'blur_sigma_in_inch' not in params or params['blur_sigma_in_inch'] is None:
        params['blur_sigma_in_inch'] = 1.0  # blurring filter sigma in inches

    if 'paper_size_in_inch' not in params or params['paper_size_in_inch'] is None:
        params['paper_size_in_inch'] = [11, 8.5]  # default paper size in inch (letter size)

    if 'remove_shadows' not in params or params['remove_shadows'] is None:
        params['remove_shadows'] = True  # remove shadows due to photography/scanning by default

    if 'apply_edge_detection' not in params or params['apply_edge_detection'] is None:
        params['apply_edge_detection'] = False  # detect grid on edge detection outputs

    if 'cluster_peaks' not in params or params['cluster_peaks'] is None:
        params['cluster_peaks'] = True  # cluster the marginal histogram peaks or not

    if params['cluster_peaks']:
        if 'max_clusters' not in params or params['max_clusters'] is None:
            params['max_clusters'] = 3  # number of clusters

        if 'cluster_selection_method' not in params or params['cluster_selection_method'] is None:
            params['cluster_selection_method'] = 'GAP_MIN_VAR'  # method for selecting clusters: 'GAP_MIN_VAR', 'MAX_AMP_PEAKS'

    if 'avg_quartile' not in params or params['avg_quartile'] is None:
        params['avg_quartile'] = 50.0  # the middle quartile used for averaging the estimated grid gaps

    if params['avg_quartile'] > 100.0:
        raise ValueError('avg_quartile parameter must be between 0 and 100.0')

    if 'sat_pre_grid_det' not in params or params['sat_pre_grid_det'] is None:
        params['sat_pre_grid_det'] = True  # saturate densities or not (before spectral estimation)

    if 'num_seg_hor' not in params or params['num_seg_hor'] is None:
        params['num_seg_hor'] = 4

    if 'num_seg_ver' not in params or params['num_seg_ver'] is None:
        params['num_seg_ver'] = 4

    if 'hist_grid_det_method' not in params or params['hist_grid_det_method'] is None:
        params['hist_grid_det_method'] = 'RANDOM_TILING'  # 'REGULAR_TILING', 'RANDOM_TILING'

    if 'total_segments' not in params or params['total_segments'] is None:
        params['total_segments'] = 100

    if 'min_grid_resolution' not in params or params['min_grid_resolution'] is None:
        params['min_grid_resolution'] = 1  # in pixels

    if 'min_grid_peak_prom_prctile' not in params or params['min_grid_peak_prom_prctile'] is None:
        params['min_grid_peak_prom_prctile'] = 2

    if 'detailed_plots' not in params or params['detailed_plots'] is None:
        params['detailed_plots'] = 0  # 0 no plots, 1 some plots, 2 all plots (for diagnosis mode only)

    width = img.shape[1]
    height = img.shape[0]

    # Convert image to grayscale if in RGB
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = img_gray.astype(np.float64) / 255.0
    else:
        img_gray = img.astype(np.float64)
        img_gray = 1.0 - img_gray / np.max(img_gray)

    # Shadow removal and intensity normalization
    if params['remove_shadows']:
        blur_sigma = np.mean([width * params['blur_sigma_in_inch'] / params['paper_size_in_inch'][0],
                              height * params['blur_sigma_in_inch'] / params['paper_size_in_inch'][1]])
        img_gray_blurred = cv2.GaussianBlur(img_gray, (0, 0), blur_sigma)
        img_gray_normalized = img_gray / img_gray_blurred
        img_gray_normalized = (img_gray_normalized - np.min(img_gray_normalized)) / (np.max(img_gray_normalized) - np.min(img_gray_normalized))
    else:
        img_gray_blurred = img_gray.copy()
        img_gray_normalized = img_gray.copy()

    # Edge detection
    if params['apply_edge_detection']:
        edges = cv2.Canny((img_gray_normalized * 255).astype(np.uint8), 50, 150)

        # Make the edges sharper
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        blur_sigma = np.mean([width * params['post_edge_det_gauss_filt_std'] / params['paper_size_in_inch'][0],
                              height * params['post_edge_det_gauss_filt_std'] / params['paper_size_in_inch'][1]])
        edges_blurred = cv2.GaussianBlur(edges.astype(np.float64), (0, 0), blur_sigma)
        edges_blurred_sat = edges_blurred.copy()

        # Saturate extreme pixels
        if params['post_edge_det_sat']:
            # Upper saturation level
            sat_level = np.percentile(edges_blurred, params['sat_level_upper_prctile'])
            edges_blurred_sat[edges_blurred_sat > sat_level] = sat_level

            # Lower saturation level
            sat_level = np.percentile(edges_blurred, params['sat_level_lower_prctile'])
            edges_blurred_sat[edges_blurred_sat < sat_level] = sat_level

        edges_blurred_sat = edges_blurred_sat / np.max(edges_blurred_sat)
        img_gray_normalized = 1.0 - (edges_blurred_sat - np.min(edges_blurred_sat)) / (np.max(edges_blurred_sat) - np.min(edges_blurred_sat))

    # Image density saturation
    if params['sat_pre_grid_det']:
        sat_level = np.percentile(img_gray_normalized, params['sat_level_pre_grid_det'])
        img_gray_normalized = np.tanh(1.0 - img_gray_normalized / sat_level)

    # Segmentation
    seg_width = width // params['num_seg_hor']
    seg_height = height // params['num_seg_ver']

    if params['hist_grid_det_method'] == 'REGULAR_TILING':
        segments_stacked = np.zeros((seg_height, seg_width, params['num_seg_hor'] * params['num_seg_ver']), dtype=np.float64)
        k = 0
        for i in range(params['num_seg_ver']):
            for j in range(params['num_seg_hor']):
                segments_stacked[:, :, k] = img_gray_normalized[i * seg_height:(i + 1) * seg_height, j * seg_width:(j + 1) * seg_width]
                k += 1
    elif params['hist_grid_det_method'] == 'RANDOM_TILING':
        segments_stacked = np.zeros((seg_height, seg_width, params['total_segments']), dtype=np.float64)
        for k in range(params['total_segments']):
            start_hor = np.random.randint(0, width - seg_width)
            start_ver = np.random.randint(0, height - seg_height)
            segments_stacked[:, :, k] = img_gray_normalized[start_ver:start_ver + seg_height, start_hor:start_hor + seg_width]

    # Horizontal/vertical histogram estimation per patch
    peak_amps_hor = []
    peak_gaps_hor = []
    peak_amps_ver = []
    peak_gaps_ver = []

    for k in range(segments_stacked.shape[2]):
        hist_hor = 1.0 - np.mean(segments_stacked[:, :, k], axis=0)
        min_grid_peak_prominence = np.percentile(hist_hor, params['min_grid_peak_prom_prctile']) - np.min(hist_hor)
        pk_amps_hor, I_pk_hor = find_peaks(hist_hor, distance=params['min_grid_resolution'], prominence=min_grid_peak_prominence)
        if len(pk_amps_hor) > 1:
            peak_amps_hor.extend(pk_amps_hor[1:])
            peak_gaps_hor.extend(np.diff(I_pk_hor))

        hist_ver = 1.0 - np.mean(segments_stacked[:, :, k], axis=1)
        min_grid_peak_prominence = np.percentile(hist_ver, params['min_grid_peak_prom_prctile']) - np.min(hist_ver)
        pk_amps_ver, I_pk_ver = find_peaks(hist_ver, distance=params['min_grid_resolution'], prominence=min_grid_peak_prominence)
        if len(pk_amps_ver) > 1:
            peak_amps_ver.extend(pk_amps_ver[1:])
            peak_gaps_ver.extend(np.diff(I_pk_ver))

    # Calculate horizontal/vertical grid sizes based on the marginal distributions with max intensity
    if not params['cluster_peaks']:  # Direct method
        peak_gaps_prctiles = np.percentile(peak_gaps_hor, [50.0 - params['avg_quartile'] / 2, 50.0 + params['avg_quartile'] / 2])
        grid_size_hor = np.mean([g for g in peak_gaps_hor if peak_gaps_prctiles[0] <= g <= peak_gaps_prctiles[1]])

        peak_gaps_prctiles = np.percentile(peak_gaps_ver, [50.0 - params['avg_quartile'] / 2, 50.0 + params['avg_quartile'] / 2])
        grid_size_ver = np.mean([g for g in peak_gaps_ver if peak_gaps_prctiles[0] <= g <= peak_gaps_prctiles[1]])
    else:  # Indirect method (cluster the local peaks)
        kmeans = KMeans(n_clusters=params['max_clusters'], random_state=0).fit(np.array(peak_amps_hor).reshape(-1, 1))
        cluster_centers_hor = kmeans.cluster_centers_
        cluster_labels_hor = kmeans.labels_

        if params['cluster_selection_method'] == 'GAP_MIN_VAR':
            peak_gaps_per_cluster = [np.std([peak_gaps_hor[i] for i in range(len(peak_gaps_hor)) if cluster_labels_hor[i] == cc]) for cc in range(params['max_clusters'])]
            selected_cluster_hor = np.argmin(peak_gaps_per_cluster)
        elif params['cluster_selection_method'] == 'MAX_AMP_PEAKS':
            peak_amps_per_cluster = [trim_mean([peak_amps_hor[i] for i in range(len(peak_amps_hor)) if cluster_labels_hor[i] == cc], proportiontocut=0.1) for cc in range(params['max_clusters'])]
            selected_cluster_hor = np.argmax(peak_amps_per_cluster)

        peak_gaps_selected_cluster_hor = [peak_gaps_hor[i] for i in range(len(peak_gaps_hor)) if cluster_labels_hor[i] == selected_cluster_hor]
        peak_gaps_prctiles = np.percentile(peak_gaps_selected_cluster_hor, [50.0 - params['avg_quartile'] / 2, 50.0 + params['avg_quartile'] / 2])
        grid_size_hor = np.mean([g for g in peak_gaps_selected_cluster_hor if peak_gaps_prctiles[0] <= g <= peak_gaps_prctiles[1]])

        kmeans = KMeans(n_clusters=params['max_clusters'], random_state=0).fit(np.array(peak_amps_ver).reshape(-1, 1))
        cluster_centers_ver = kmeans.cluster_centers_
        cluster_labels_ver = kmeans.labels_

        if params['cluster_selection_method'] == 'GAP_MIN_VAR':
            peak_gaps_per_cluster = [np.std([peak_gaps_ver[i] for i in range(len(peak_gaps_ver)) if cluster_labels_ver[i] == cc]) for cc in range(params['max_clusters'])]
            selected_cluster_ver = np.argmin(peak_gaps_per_cluster)
        elif params['cluster_selection_method'] == 'MAX_AMP_PEAKS':
            peak_amps_per_cluster = [trim_mean([peak_amps_ver[i] for i in range(len(peak_amps_ver)) if cluster_labels_ver[i] == cc], proportiontocut=0.1) for cc in range(params['max_clusters'])]
            selected_cluster_ver = np.argmax(peak_amps_per_cluster)

        peak_gaps_selected_cluster_ver = [peak_gaps_ver[i] for i in range(len(peak_gaps_ver)) if cluster_labels_ver[i] == selected_cluster_ver]
        peak_gaps_prctiles = np.percentile(peak_gaps_selected_cluster_ver, [50.0 - params['avg_quartile'] / 2, 50.0 + params['avg_quartile'] / 2])
        grid_size_ver = np.mean([g for g in peak_gaps_selected_cluster_ver if peak_gaps_prctiles[0] <= g <= peak_gaps_prctiles[1]])

    # Plot results
    if params['detailed_plots'] > 0:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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

        plt.suptitle('Preprocessing stages (shadow removal and intensity normalization)')
        plt.show()

    return grid_size_hor, grid_size_ver
