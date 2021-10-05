"""Functions to create Fourier-space filters."""

def __x2y2(w, h):
    """Creates the mesh grid of x^2 + y^2 from -w/2 to w/2 and -h/2 to h/2"""
    # Actually uses ogrid (open-grid) instead of meshgrid() since it is more efficient and less code
    from numpy import ogrid
    x,y = ogrid[-(w//2):((w+1)//2), -(h//2):((h+1)//2)]
    return y*y + x*x


def ideal_low_pass(w, h, D):
    """
    Creates a Fourier-space ideal low-pass filter of the given width and height with the cutoff D.
    """
    return (__x2y2(w,h)<=(D*D)).astype(float)


def ideal_high_pass(w, h, D):
    """
    Creates a Fourier-space ideal high-pass filter of the given width and height with the cutoff D.
    """
    return (__x2y2(w,h)>(D*D)).astype(float)


def butterworth_low_pass(w, h, D, n):
    """
    Creates a Fourier-space Butterworth low-pass filter of the given width and height with the
    cutoff D and order n.
    """
    return 1 / (1 + (__x2y2(w,h)/(D*D))**n)


def butterworth_high_pass(w, h, D, n):
    """
    Creates a Fourier-space Butterworth high-pass filter of the given width and height with the
    cutoff D and order n.
    """
    return 1 - butterworth_low_pass(w, h, D, n)


def gaussian(w, h, sigma, normed=False):
    """
    Creates a Gaussian centered in a w x h image with the given standard deviation sigma. By
    default this has a peak of 1. If normed is True then this is normalized so it sums to 1.
    """
    from numpy import exp
    g = exp(-__x2y2(w,h)/(sigma*sigma))
    if normed: g /= g.sum()
    return g


def gaussian_low_pass(w, h, sigma):
    """
    Creates a Gaussian low-pass filter centered in a w x h image with the given standard deviation
    sigma.
    """
    return gaussian(w, h, sigma)


def gaussian_high_pass(w, h, sigma):
    """
    Creates a Gaussian high-pass filter centered in a w x h image with the given standard deviation
    sigma.
    """
    return 1 - gaussian(w, h, sigma)
