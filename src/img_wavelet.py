import numpy as np
import math

def corr_1d(signal, kernel):
    """'same' correlation with periodic padding"""
    n = kernel.shape[0]
    spaces = math.ceil((n-1)/2)

    if n % 2 != 0:
        padded = np.pad(signal, (spaces, spaces), mode='wrap')
    else:
        padded = np.pad(signal, (spaces, spaces-1), mode='wrap')
    result = np.correlate(padded, kernel, mode='valid') # Still, 'same' through padding
    return result

def dwt_1d(sig, levels, scaling_function, wavelet_function):
    """
    Returns a list of np.arrays. [approx., detail]
    If legnth of signal do not allow to reach all levels, truncate 
    """
    # Check if odd
    if len(sig) % 2 != 0:
        sig = np.append(sig, sig[0]) # alternative way?
    
    X = []
    for i in range(levels):
        # "double inversion" means simple correlation
        approx = corr_1d(sig, scaling_function)
        detail = corr_1d(sig, wavelet_function)
        # Decimate (start at 1 to match pywt)
        approx = approx[1::2]
        detail = detail[1::2]
        X.append({"approximation": approx, "detail": detail})

        # Peparation of the next loop
        sig = approx.copy()
        if len(sig) % 2 != 0:
            sig = np.append(sig, 0)

        if len(approx) <= len(scaling_function):
            break
    return X

def img_wavelet(img):
    ''' Returns 1st level "Blur" 2d wavelet of img.
    The output is downsized by 2 if dimesion is even.
    If it is odd, output dimension is downsized by 2 and
    rounded up by one. Handles non-square images.
    
    Implementation from chp7, digital image processing, 
    Gonzales 4th edition.'''
    rows, cols = img.shape

    # Using haar wavelets
    scaling_function = (1/np.sqrt(2)) * np.array([1, 1])
    wavelet_function =  (1/np.sqrt(2)) * np.array([1, -1])

    cols_new = math.ceil(cols/2)
    img_row_dwt = np.zeros((rows, cols_new))
    for row in range(rows):
        img_row_dwt[row,:] = dwt_1d(img[row,:], 1, 
                                    scaling_function, 
                                    wavelet_function)[0]['approximation']

    rows_new = math.ceil(rows/2)
    img_col_dwt = np.zeros((rows_new, cols_new))
    for col in range(cols_new):
        img_col_dwt[:,col] = dwt_1d(img_row_dwt[:,col], 1, 
                                    scaling_function, 
                                    wavelet_function)[0]['approximation']
    
    return img_col_dwt










