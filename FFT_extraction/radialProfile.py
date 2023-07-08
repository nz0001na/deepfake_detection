# from https://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
import numpy as np

def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image,
    # y is indices of row (same one row), x is indices of col (same one col)
    y, x = np.indices(image.shape)

    # center is calculated based on the col of the img, 0.5*num_of_col
    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    # r is symmetric
    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat) # flat r, then sort it and get the sort index_r
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]  # sort image by index_r

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented, cuo wei xiang jian
    rind = np.where(deltar)[0]       # location of changed radius , fei 0 chu is changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin, #### is related to img size####
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof
