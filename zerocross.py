def zerocross(im):
    """Finds the zero-crossing in the given image, returning a binary image."""
    from numpy import pad
    p = im[1:-1,1:-1]
    u,d = im[:-2,1:-1], im[2:,1:-1]
    l,r = im[1:-1,:-2], im[1:-1,2:]
    out = (((p<0)&((u>0)|(r>0)|(d>0)|(l>0))) |
           ((p==0)&(((u<0)^(d<0))|((r<0)^(l<0)))))
    return pad(out, 1, 'constant')
