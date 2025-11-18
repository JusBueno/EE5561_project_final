import numpy as np
import cvxpy as cp

def transform_downsample(img, percentage):
    '''Keep just percentage of rows in frequency domain'''
    rows, cols = img.shape
    rows_to_keep = int(np.ceil(percentage*rows))
    img_fourier = np.fft.fftshift(np.fft.fft2(img))
    
    rows_list = np.arange(rows)
    np.random.shuffle(rows_list)
    rows_shuffled = rows_list[:rows_to_keep]

    img_fourier_down = np.ones((rows, cols), dtype=complex) * 1e-15
    mask = np.zeros((rows, cols), dtype=complex)

    for row in range(rows_to_keep):
        img_fourier_down[rows_shuffled[row],:] = img_fourier[rows_shuffled[row],:]
        mask[rows_shuffled[row],:] = 1

    return img_fourier_down, mask

def transform_downsample_reconstruct(img, percentage):
    '''Generate degradated and then reconstructed images'''
    rows, cols = img.shape
    img_fourier_down, mask = transform_downsample(img, percentage)

    # Reverse the shift for compatilibility with transform
    img_fourier_down = np.fft.ifftshift(img_fourier_down)
    mask = np.fft.ifftshift(mask)

    # Tranform, matrix form
    F = np.fft.fft(np.eye(rows),norm='ortho')

    # Minimization problem (inspired from the cvxpy image inpainting tv problem example)
    X = cp.Variable(shape=(rows, cols))
    obj = cp.Minimize(cp.tv(X))
    constraints = [cp.multiply(mask,F@X@F.T) == cp.multiply(mask,img_fourier_down)]
    prob = cp.Problem(obj, constraints)

    # Use SCS to solve the problem.
    prob.solve(solver=cp.SCS)
    return X.value





