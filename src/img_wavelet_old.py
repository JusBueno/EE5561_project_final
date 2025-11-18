import pywt
import numpy as np
from correlate_2d import correlate_2d

def img_wavelet(img, level):
    wv_coef = pywt.wavedec2(img, 'haar', level=level)
    return wv_coef[0]

