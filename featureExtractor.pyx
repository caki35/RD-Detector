import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt
from skimage.measure import label


cpdef cnp.ndarray[cnp.uint8_t, ndim=2] moveToLeft(cnp.ndarray[cnp.uint8_t, ndim=2] binary_img, int tx):
    cdef int y, x
    cdef int rows = binary_img.shape[0]
    cdef int cols = binary_img.shape[1]
    
    # Prepare output image
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] translated_img = np.zeros_like(binary_img)
    
    # Iterate over the image with C-style loops for speed
    for y in range(rows):
        for x in range(cols):
            if binary_img[y, x] == 1:
                new_x = x - tx
                if 0 <= new_x < cols:
                    translated_img[y, new_x] = 1
                    
    return translated_img

# Ensure the data type is explicitly declared for speed optimization
def getLargestCC(cnp.ndarray[cnp.uint8_t, ndim=2] segmentation):
    cdef cnp.ndarray labels = label(segmentation)
    cdef int max_label = labels.max()
    assert max_label != 0  # Ensure at least one connected component exists
    
    # Compute the largest connected component
    cdef cnp.ndarray[cnp.intp_t] counts = np.bincount(labels.ravel())[1:]
    cdef int largest_label = np.argmax(counts) + 1
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] largestCC = (labels == largest_label).astype(np.uint8)
    
    return largestCC


# Main filtering function with Cython optimization
cpdef cnp.ndarray[cnp.uint8_t, ndim=2] filterRetina(cnp.ndarray[cnp.uint8_t, ndim=2] pred):
    cdef int y_center, x_center, tx
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] binary_img
    cdef cnp.ndarray[cnp.intp_t] y_indices, x_indices
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] translated_image

    # Create binary image where `pred == 3`
    binary_img = np.zeros_like(pred)
    binary_img[pred == 3] = 1

    # Get the largest connected component using the optimized Cython function
    binary_img = getLargestCC(binary_img)

    # Calculate y-center
    y_indices = np.where(binary_img == 1)[0]
    y_center = int(np.mean(y_indices))

    # Calculate the center row and how far to shift left
    x_indices = np.where(binary_img[y_center, :] == 1)[0]
    x_center = int(np.mean(x_indices))
    tx = len(x_indices)

    # Translate the image
    translated_image = moveToLeft(binary_img, tx)

    # Remove overlapping pixels by zeroing them out
    pred[translated_image == 1] = 0
    return pred

cpdef cnp.ndarray[cnp.float64_t, ndim=2] featureExtract(cnp.ndarray[cnp.uint8_t, ndim=2] segMap):
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] RetinaArea = (segMap == 2).astype(np.uint8)    
    cdef cnp.ndarray[cnp.int_t, ndim=1] indices
    cdef cnp.ndarray[cnp.int_t, ndim=1] diff
    cdef list projectedList = []

    cdef int c
    for c in range(RetinaArea.shape[1]):
        if (np.sum(RetinaArea[:,c]) != 0):
            indices = np.where(RetinaArea[:,c] ==1)[0]
            diff = np.diff(indices)
            projectedList.append(np.count_nonzero(diff != 1)+1)
            
    cdef double projected1D_max = max(projectedList)
    cdef double projected1D_std = np.std(projectedList)

    cdef cnp.ndarray[cnp.uint8_t, ndim=2] segimg_filtered = filterRetina(segMap.copy())
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] binary_img_retina = np.zeros_like(segimg_filtered)
    binary_img_retina[segimg_filtered == 2] = 1
    cdef double areaRetina = np.sum(binary_img_retina)


    # Original 1D feature vector
    cdef cnp.ndarray[cnp.float64_t, ndim=2] feature_vector_2D = np.array(
        [projected1D_std, projected1D_max, areaRetina], dtype=np.float64
    ).reshape(1, -1)

    return feature_vector_2D