import argparse
import cv2
import numpy as np
import numba as nb


EIGHT_CONNECTED_NEIGHBOR_KERNEL = np.array([[1., 1., 1.],
                                            [1., 0., 1.],
                                            [1., 1., 1.]], dtype=np.float64)
SIGMA_COEFF = 6.4      # The denominator for a 2D Gaussian sigma used in the reference implementation.
ERROR_THRESHOLD = 0.2  # The default error threshold for synthesis acceptance in the reference implementation.

LONG_M = np.ones((16, 16)) * 10.0

def normalized_ssd(sample, window, mask):
    wh, ww = window.shape
    sh, sw = sample.shape

    # Get sliding window views of the sample, window, and mask.
    strided_sample = np.lib.stride_tricks.as_strided(sample, shape=((sh-wh+1), (sw-ww+1), wh, ww), 
                        strides=(sample.strides[0], sample.strides[1], sample.strides[0], sample.strides[1]))
    strided_sample = strided_sample.reshape(-1, wh, ww)

    for i in range(strided_sample.shape[0]):
        hole = np.where(strided_sample[i] == 1.)
        if len(hole[0]) != 0:
            strided_sample[i] = np.copy(LONG_M)

    # Note that the window and mask views have the same shape as the strided sample, but the kernel is fixed
    # rather than sliding for each of these components.
    strided_window = np.lib.stride_tricks.as_strided(window, shape=((sh-wh+1), (sw-ww+1), wh, ww),
                        strides=(0, 0, window.strides[0], window.strides[1]))
    strided_window = strided_window.reshape(-1, wh, ww)
    # strided_window = strided_window[0:strided_sample.shape[0]]


    strided_mask = np.lib.stride_tricks.as_strided(mask, shape=((sh-wh+1), (sw-ww+1), wh, ww),
                        strides=(0, 0, mask.strides[0], mask.strides[1]))
    strided_mask = strided_mask.reshape(-1, wh, ww)
    # strided_mask = strided_mask[0:strided_sample.shape[0]]


    # Form a 2D Gaussian weight matrix from symmetric linearly separable Gaussian kernels and generate a 
    # strided view over this matrix.
    sigma = wh / SIGMA_COEFF
    kernel = cv2.getGaussianKernel(ksize=wh, sigma=sigma)
    kernel_2d = kernel * kernel.T

    strided_kernel = np.lib.stride_tricks.as_strided(kernel_2d, shape=((sh-wh+1), (sw-ww+1), wh, ww),
                        strides=(0, 0, kernel_2d.strides[0], kernel_2d.strides[1]))
    strided_kernel = strided_kernel.reshape(-1, wh, ww)
    # strided_kernel = strided_kernel[0:strided_sample.shape[0]]

    # Take the sum of squared differences over all sliding sample windows and weight it so that only existing neighbors
    # contribute to error. Use the Gaussian kernel to weight central values more strongly than distant neighbors.
    squared_differences = ((strided_sample - strided_window)**2) * strided_kernel * strided_mask
    ssd = np.sum(squared_differences, axis=(1,2))
    ssd = ssd.reshape(sh-wh+1, sw-ww+1)

    # Normalize the SSD by the maximum possible contribution.
    total_ssd = np.sum(mask * kernel_2d)
    normalized_ssd = ssd / total_ssd

    return normalized_ssd

def get_candidate_indices(normalized_ssd, error_threshold=ERROR_THRESHOLD):
    min_ssd = np.min(normalized_ssd)
    min_threshold = min_ssd * (1. + error_threshold)
    indices = np.where(normalized_ssd <= min_threshold)
    return indices

def select_pixel_index(normalized_ssd, indices, method='uniform'):
    N = indices[0].shape[0]

    if method == 'uniform':
        weights = np.ones(N) / float(N)
    else:
        weights = normalized_ssd[indices]
        weights = weights / np.sum(weights)

    # Select a random pixel index from the index list.
    selection = np.random.choice(np.arange(N), size=1, p=weights)
    selected_index = (indices[0][selection], indices[1][selection])
    
    return selected_index

def get_neighboring_pixel_indices(pixel_mask):
    # Taking the difference between the dilated mask and the initial mask
    # gives only the 8-connected neighbors of the mask frontier.
    kernel = np.ones((4, 4))
    dilated_mask = cv2.dilate(pixel_mask, kernel, iterations=1)
    neighbors = dilated_mask - pixel_mask

    # Recover the indices of the mask frontier.
    neighbor_indices = np.nonzero(neighbors)

    return neighbor_indices

def permute_neighbors(pixel_mask, neighbors):
    N = neighbors[0].shape[0]

    # Generate a permutation of the neigboring indices
    permuted_indices = np.random.permutation(np.arange(N))
    permuted_neighbors = (neighbors[0][permuted_indices], neighbors[1][permuted_indices])

    # Use convolution to count the number of existing neighbors for all entries in the mask.
    neighbor_count = cv2.filter2D(pixel_mask, ddepth=-1, kernel=EIGHT_CONNECTED_NEIGHBOR_KERNEL, borderType=cv2.BORDER_CONSTANT)

    # Sort the permuted neighboring indices by quantity of existing neighbors descending.
    permuted_neighbor_counts = neighbor_count[permuted_neighbors]

    sorted_order = np.argsort(permuted_neighbor_counts)[::-1]
    permuted_neighbors = (permuted_neighbors[0][sorted_order], permuted_neighbors[1][sorted_order])

    return permuted_neighbors

def texture_can_be_synthesized(mask):
    # The texture can be synthesized while the mask has unfilled entries.
    mh, mw = mask.shape[:2]
    num_completed = np.count_nonzero(mask)
    num_incomplete = (mh * mw) - num_completed
    
    return num_incomplete > 0

def initialize_texture_synthesis(original_sample, kernel_size):
    # Convert original to sample representation.
    sample = cv2.cvtColor(original_sample, cv2.COLOR_BGR2GRAY)
    
    # Convert sample to floating point and normalize to the range [0., 1.]
    sample = sample.astype(np.float64)
    sample = sample / 255.

    # Generate window
    window = np.copy(sample)

    # Generate output window
    result_window = np.copy(original_sample)

    # Generate window mask
    h, w = window.shape
    mask = np.ones((h, w), dtype=np.float64)

    hole_index = np.where(sample == 1.)

    for hole_x, hole_y in zip(hole_index[0], hole_index[1]):
        window[hole_x, hole_y] = 0.
        mask[hole_x, hole_y] = 0

    return sample, window, mask, result_window

def synthesize_texture(original_sample, kernel_size, visualize):
    global gif_count

    half_size = kernel_size // 2

    (sample, window, mask, result_window) = initialize_texture_synthesis(original_sample, kernel_size)

    # Synthesize texture until all pixels in the window are filled.
    while texture_can_be_synthesized(mask):
        # Get neighboring indices
        neighboring_indices = get_neighboring_pixel_indices(mask)

        # Permute and sort neighboring indices by quantity of 8-connected neighbors.
        neighboring_indices = permute_neighbors(mask, neighboring_indices)
        # print(neighboring_indices)
        
        for ch, cw in zip(neighboring_indices[0], neighboring_indices[1]):
            # print(ch, cw)

            window_slice = window[ch-half_size:ch+half_size, cw-half_size:cw+half_size]
            mask_slice = mask[ch-half_size:ch+half_size, cw-half_size:cw+half_size]

            # Compute SSD for the current pixel neighborhood and select an index with low error.
            ssd = normalized_ssd(sample, window_slice, mask_slice)
            # print(ssd)
            indices = get_candidate_indices(ssd)
            selected_index = select_pixel_index(ssd, indices)

            # Translate index to accommodate padding.
            selected_index = (selected_index[0] + half_size, selected_index[1] + half_size)

            # Set windows and mask.
            window[ch, cw] = sample[selected_index]
            mask[ch, cw] = 1
            result_window[ch, cw] = original_sample[selected_index[0], selected_index[1]]

            if visualize:
                cv2.imshow('synthesis window', result_window)
                key = cv2.waitKey(1) 
                if key == 27:
                    cv2.destroyAllWindows()
                    return result_window

    if visualize:
        cv2.imshow('synthesis window', result_window)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result_window


