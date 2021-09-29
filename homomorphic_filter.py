def homomorphic_filter(im, cutoff, order=2, lowgain=0.5, highgain=2):
    """
    Applies a homomorphic filter to an image using a Butterworth filter as the
    low-pass filter base for the high-boost filter.
    """
    from numpy import log, ogrid, exp
    from scipy import fft
    im = im.astype(float)
    im[im==0] = 1 # prevent taking the log of 0
    lg = log(im)
    ft = fft.fftshift(fft.fft2(lg))
    h,w = im.shape
    y,x = ogrid[-(h//2):(h+1)//2, -(w//2):(w+1)//2]  # similar to meshgrid()
    bw_fltr = 1/(1+0.414*((x*x+y*y)/(cutoff*cutoff))**order)
    fltr = lowgain + (highgain - lowgain) * (1 - bw_fltr)
    fltred = fltr * ft
    out = exp(fft.ifft2(fft.ifftshift(fltred)).real)
    # rescale the intensities
    out -= out.min()
    out *= 255/out.max()
    return out.clip(0, 255).astype('uint8')
