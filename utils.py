"""General utility functions."""

import numpy as np
from numpy import fft

def nonzero(x):
    """
    If given 0 then this returns an extremely tiny, but non-zero, positive value. Otherwise the
    given value is returned. The value x can be a scalar or an array.
    """
    if not isinstance(x, np.ndarray):
        return np.finfo(float).eps if x == 0 else x
    x[x == 0] = np.finfo(float).eps
    return x


def psf2otf(psf, shape):
    """
    Convert a PSF to an OTF. This is essentially fft.fft2(psf, shape) except it also includes a
    minor shift of the data so that the center of the PSF is at (0,0) before the Fourier transform
    is computed but after padding.
    """
    psf_shape = psf.shape
    psf = np.pad(psf, ((0, shape[0] - psf_shape[0]), (0, shape[1] - psf_shape[1])), 'constant')
    psf = np.roll(psf, (-(psf_shape[0]//2), -(psf_shape[1]//2)), axis=(0,1)) # shift PSF so center is at (0,0)
    return fft.fft2(psf)


def otf2psf(otf, shape):
    """
    Convert an OTF to a PSF. This is essentially fft.ifft2(otf, shape) except it also includes a
    minor shift of the data so that the center of the PSF is moved back to the middle after the
    inverse Fourier transform is computed but before cropping.
    """
    psf = fft.ifft2(otf)
    psf = np.roll(psf, (shape[0]//2, shape[1]//2), axis=(0,1)) # shift PSF so center is in middle
    return psf[:shape[0], :shape[1]]
