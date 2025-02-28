from skimage.measure import label
import numpy as np
cimport numpy as cnp

cpdef cnp.ndarray[cnp.uint8_t, ndim=2] NoiseFiltering(cnp.ndarray[cnp.uint8_t, ndim=2] img, int thresh=1500):
    """
    Perform noise filtering on the input image.
    
    Parameters:
        img: 2D numpy array of uint8 type
        thresh: Minimum threshold for connected component size
    
    Returns:
        img: Modified image with small connected components removed
    """
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] binary_img = np.zeros_like(img)
    binary_img[img == 2] = 1

    cdef cnp.ndarray[cnp.int32_t, ndim=2] label_img = label(binary_img).astype(np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] label_list = np.unique(label_img)

    cdef int lbl
    for lbl in label_list:
        if np.sum(label_img == lbl) < thresh:
            img[label_img == lbl] = 0

    return img