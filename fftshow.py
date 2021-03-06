import numpy as np

def fftshow(arr, mode='mag', log_scale=True, eliminate_dc=True, plot=True):
    """
    Shows a 2D Fourier transform using one of the following modes:
      * "mag" - the complex magnitude (default)
      * "color" - map complex numbers onto a color wheel using HSV color model
      
    For color mapping the complex angle is the hue (red is pos. real, cyan is
    neg. real, yellow/green is pos. imaginary, and purple is neg. imaginary),
    the saturation is 1, and the values are the complex magnitude; some hue
    adjustments are made to neighboring angles that are 180 from each other.
    
    By default the DC component of the image is eliminated and the magnitude
    data is log-scaled to make them displayable relative to each other. Set
    `log_scale` and `eliminate_dc` to False to turn these off.
    
    Instead of plotting the image the generated image can be returned so that
    it can be saved or otherwise manipulated.
    """
    # Calculate the complex magnitude
    mag = abs(arr)
    
    # Find the DC component
    dc_i, dc_j = np.unravel_index(mag.argmax(), mag.shape)
    # Eliminate the DC component
    if eliminate_dc:
        mag[dc_i,dc_j] = 0 # clear it
        mag[dc_i,dc_j] = mag[dc_i,dc_j].max() # set it to the second-max

    # Logarithm scaling
    if log_scale: mag = np.log(mag+1)
    
    # Map image data
    if mode == 'mag': im = mag # Just magnitude
    elif mode == 'color':
        # Color mapping using HSV
        # Hue is based on the angle of the complex number
        H = np.arctan2(arr.imag, arr.real) # Calculate the complex angle (-pi to pi)
        H[::2,::2] += np.pi; H[1::2,1::2] += np.pi # adjust for cross-hatch pattern of angles
        H /= 2*np.pi; H %= 1 # convert from -pi to pi range to 0 to 1
        H[dc_i, dc_j] = 0 # correct DC component angle
        
        # The saturation is all 1s (except DC)
        S = np.ones(arr.shape)
        S[dc_i, dc_j] = 0
        
        # The values are the normalized magnitude of the complex number (from 0 to 1)
        mag -= mag.min(); mag /= mag.max(); V = mag

        # Convert HSV to RGB for displaying
        from matplotlib.colors import hsv_to_rgb
        im = hsv_to_rgb(np.dstack((H, S, V)))
    
    # Unknown mode
    else: raise ValueError('mode must be one of "mag" or "color"')
        
    # Show or return the image
    if plot:
        from matplotlib.pylab import imshow
        imshow(im)
    else: return im
