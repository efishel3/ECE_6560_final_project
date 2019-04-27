

# ECE 6560 Final Project
# Eydan Fishel
# Horn-Schunck Algorithm Helper Function

import numpy as np
from scipy import signal


def horn_schunck(u, v, Ix, Iy, It, param, alpha):
    # Set constants
    avg_kernel = np.array([[1/12,  1/6, 1/12],
                           [1/6,    0,  1/6],
                           [1/12,  1/6, 1/12]])
    if param is None:
        num_iterations = 100
    else:
        num_iterations = param
    for i in range(num_iterations):
        # Iteratively filter the u and v arrays with averaging filters
        u_avg = signal.convolve2d(u, avg_kernel, mode='same')
        v_avg = signal.convolve2d(v, avg_kernel, mode='same')

    # Compute flow vectors constrained by local averages and the optical flow constraints
    u = u_avg - ((Ix*(Ix*u_avg + Iy*v_avg + It))/((Ix**2) + (Iy**2) + (alpha**(-1))))
    v = v_avg - ((Iy*(Ix*u_avg + Iy*v_avg + It))/((Ix**2)+(Iy**2)+(alpha**(-1))))

    # Compute and display the nan and inf ratios
    u_nan_ratio = float(np.sum(np.isnan(u)) / u.size)
    v_nan_ratio = float(np.sum(np.isnan(v)) / v.size)
    u_inf_ratio = float(np.sum(np.isinf(u)) / u.size)
    v_inf_ratio = float(np.sum(np.isinf(v)) / v.size)
    print('Estimated Flow nan ratio: u = {:.2f}, v = {:.2f}'
          .format(u_nan_ratio, v_nan_ratio))
    print('Estimated Flow inf ratio: u = {:.2f}, v = {:.2f}'
          .format(u_inf_ratio, v_inf_ratio))
    # Remove nan values from u and v
    u[np.isnan(u)] = 0
    v[np.isnan(v)] = 0

    return u, v

