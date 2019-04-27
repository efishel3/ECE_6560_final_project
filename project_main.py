
# ECE 6560 Final Project
# Eydan Fishel
# April 27, 2019

"""Use optical flow constraints to determine a range of values for parameter lambda that make two images converge"""

import cv2
from scipy import signal
import numpy as np
from alg import horn_schunck
from timeit import default_timer
from matplotlib import pyplot as plt
from utils import read_flow_file

filepaths = ('images/Drop/',
             'images/Urban/')


def run_optical_flow(filepath_ind: int, param: int=500,
                     display: bool=True):

    frame_1 = cv2.imread(filepaths[filepath_ind] + 'frame1.png')[:, :, :3]
    frame_2 = cv2.imread(filepaths[filepath_ind] + 'frame2.png')[:, :, :3]

    frame_1_gray = cv2.cvtColor(frame_1, cv2.COLOR_RGB2GRAY)
    frame_2_gray = cv2.cvtColor(frame_2, cv2.COLOR_RGB2GRAY)

    # Convert to grayscale for analysis
    frame_1 = frame_1_gray.astype(float)
    frame_2 = frame_2_gray.astype(float)

    start_time = default_timer()

    if display:
        # Plot the images
        plt.figure()
        plt.imshow(frame_1, 'gray')
        plt.figure()
        plt.imshow(frame_2, 'gray')

    # Initialize kernels for finding partial derivatives of the image
    kernelx = np.array([[-1, 1], [-1, 1]])
    kernely = np.array([[-1, -1], [1, 1]])
    kernelt_1 = np.array([[1, 1], [1, 1]])
    kernelt_2 = np.array([[-1, -1], [-1, -1]])
    Ix = signal.convolve2d(frame_1, kernelx, mode='same')
    Iy = signal.convolve2d(frame_2, kernely, mode='same')
    It = signal.convolve2d(frame_1, kernelt_1, mode='same') + signal.convolve2d(frame_2, kernelt_2, mode='same')

    # Create empty u and v matrices for recursion
    u = np.zeros_like(frame_1)
    v = np.zeros_like(frame_1)

    # Alpha is the regulatory parameter
    alpha_vals = np.array([1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2])
    RMSE = np.zeros_like(alpha_vals)
    count = 0
    for alpha in alpha_vals:
        u, v = horn_schunck(u, v, Ix, Iy, It, param, alpha)
        end_time = default_timer()

        # Determine run time
        duration = end_time - start_time
        clock = [int(duration // 60), int(duration % 60)]
        print('Flow estimation time was {} minutes and {} seconds'
                .format(*clock))

        # Downsample for better visuals and results
        stride = 10
        m, n = frame_1.shape
        x, y = np.meshgrid(range(n), range(m))
        x = x.astype('float64')
        y = y.astype('float64')

        # Downsampled u and v
        u_ds = u[::stride, ::stride]
        v_ds = v[::stride, ::stride]

        # Coordinates for downsampled u and v
        x_ds = x[::stride, ::stride]
        y_ds = y[::stride, ::stride]

        # Estimated flow
        estimated_flow = np.stack((u, v), axis=2)

        # Read file for ground truth flow
        ground_truth_flow = read_flow_file(filepaths[filepath_ind] + 'flow1_2.flo')
        u_gt_orig = ground_truth_flow[:, :, 0]
        v_gt_orig = ground_truth_flow[:, :, 1]
        u_gt = np.where(np.isnan(u_gt_orig), 0, u_gt_orig)
        v_gt = np.where(np.isnan(v_gt_orig), 0, v_gt_orig)

        # Downsampled u_gt and v_gt
        u_gt_ds = u_gt[::stride, ::stride]
        v_gt_ds = v_gt[::stride, ::stride]

        RMSE[count] = np.sqrt(np.sum(((u_ds - u_gt_ds) ** 2) + ((v_ds - v_gt_ds) ** 2)))
        print(RMSE)
        count = count + 1
    plt.figure()
    plt.semilogx(alpha_vals, RMSE)
    plt.xlabel('Lambda Values')
    plt.ylabel('Root Mean Squared Error')
    plt.title('RMSE for Range of Lambda Values')

    if display:
        # Plot the optical flow field
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(frame_2, 'gray')
        plt.quiver(x_ds, y_ds, u_ds, v_ds, color='r')
        plt.title('Estimated', fontsize='x-small')
        plt.subplot(1, 2, 2)
        plt.imshow(frame_2, 'gray')
        plt.quiver(x_ds, y_ds, u_gt_ds, v_gt_ds, color='r')
        plt.title('Ground Truth', fontsize='x-small')
        """
        # Draw colored velocity flow maps
        plt.subplot(2, 2, 3)
        plt.imshow(flow_to_color(estimated_flow))
        plt.title('Estimated', fontsize='x-small')
        plt.subplot(2, 2, 4)
        plt.imshow(flow_to_color(ground_truth_flow))
        plt.title('Ground Truth', fontsize='x-small')
        """
    # Normalization for metric computations
    normalize = lambda im: (im - np.min(im)) / (np.max(im) - np.min(im))
    un = normalize(u)
    un_gt = normalize(u_gt)
    un_gt[np.isnan(u_gt_orig)] = 1
    vn = normalize(v)
    vn_gt = normalize(v_gt)
    vn_gt[np.isnan(v_gt_orig)] = 1

    # Error calculations and displays
    EPE = ((un - un_gt) ** 2 + (vn - vn_gt) ** 2) ** 0.5
    AE = np.arccos(((un * un_gt) + (vn * vn_gt) + 1) /
                   (((un + vn + 1) * (un_gt + vn_gt + 1)) ** 0.5))
    EPE_nan_ratio = np.sum(np.isnan(EPE)) / EPE.size
    AE_nan_ratio = np.sum(np.isnan(AE)) / AE.size
    EPE_inf_ratio = np.sum(np.isinf(EPE)) / EPE.size
    AE_inf_ratio = np.sum(np.isinf(AE)) / AE.size
    print('Error nan ratio: EPE={:.2f}, AE={:.2f}'
            .format(EPE_nan_ratio, AE_nan_ratio))
    print('Error inf ratio: EPE={:.2f}, AE={:.2f}'
            .format(EPE_inf_ratio, AE_inf_ratio))
    EPE_avg = np.mean(EPE[~np.isnan(EPE)])
    AE_avg = np.mean(AE[~np.isnan(AE)])
    print('EPE={:.2f}, AE={:.2f}'.format(EPE_avg, AE_avg))

    if display:
        plt.show()

    return clock, EPE_avg, AE_avg


if __name__ == '__main__':
    # Set this variable to choose which image set to use
    # 0 - Drop
    # 1 - Urban
    filepath_ind = 1

    run_optical_flow(filepath_ind)
