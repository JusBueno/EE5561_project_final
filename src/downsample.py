import numpy as np

def correlate_2d(img, kernel):
    """Same output size 2D correlation using zero-padding"""
    m, n = img.shape
    p, q = kernel.shape # Tested for odd kernels

    # zero-padd the signal to get same output size
    padded = np.zeros((m+p-1, n+q-1))
    padded[:m, :n] = img
    padded = np.roll(padded, (p-1)//2, axis=0)
    padded = np.roll(padded, (q-1)//2, axis=1)

    # create indices to extract all pxq patches centered on each (i,j) pixel. 
    # Automatically handles output size (same as input size).
    idx_row = np.array(np.arange(m)[:,None] + np.arange(p)[None,:]).flatten()
    idx_col = np.array(np.arange(n)[:,None] + np.arange(q)[None,:]).flatten()

    # create a matrix of matrices (4D array), where each (i,j) matrix is the pxq patch
    # associated with with that (i,j) pixel.
    blocks = padded[np.ix_(idx_row,idx_col)].reshape(m,p,n,q)
    blocks_transpose = np.transpose(blocks,(0,2,1,3)) # turn it into block matrix

    # correlation operation is just the product of the kernel with each individual (i,j)
    # matrix/patch (adjust dimension of kernel to allow numpy broadcasting), and the sum
    # of such products for each patch.
    block_product = blocks_transpose * kernel[None, None, :, :]
    correlation = np.sum(np.sum(block_product, axis=-1), axis=-1) # respects summation order.
    return correlation

def downsample(img, factor):
    return img[::factor, ::factor]

def gaussian_kernel(kernel_size, sigma):
    x = np.reshape(np.arange(-kernel_size//2 + 1, kernel_size//2 + 1), (-1,1))
    gaussian_1d = np.exp(-0.5 * (x / sigma)**2)

    # By the separability principle of gaussian kernel
    gaussian_2d = gaussian_1d @ gaussian_1d.T
    return gaussian_2d / np.sum(gaussian_2d)

def filter_downsample(img, factor):
    '''Antialias filtered image before donwsampling'''
    sigma = 2 * factor / 6.0 # from scikit-image documentation
    kernel_size = 2 * factor + 1 # to keep kernel size odd
    kernel = gaussian_kernel(kernel_size, sigma)
    img_filtered = correlate_2d(img, kernel)
    return img_filtered[::factor, ::factor]