
import numpy as np
from scipy.spatial.distance import euclidean, cosine
import cv2


# Reference: https://www.geeksforgeeks.org/measure-similarity-between-images-using-python-opencv/


# convert PIL image to OpenCV image: https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
def convert_PIL_to_opencv(pilimg):
    npimg = np.array(pilimg.convert('RGB'))
    cvimg = npimg[:, :, ::-1].copy()
    return cvimg


def compute_grayscale_histogram(cvimg, normalize=False):
    grayimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([grayimg], [0], None, [256], [0, 256])
    if normalize:
        return hist / np.linalg.norm(hist)
    else:
        return hist


def get_histvector_from_PIL(pilimg, normalize=False):
    cvimg = convert_PIL_to_opencv(pilimg)
    histvec = compute_grayscale_histogram(cvimg, normalize=normalize)
    return histvec


def imagepair_distance(img1, img2, normalize=False):
    hist1 = get_histvector_from_PIL(img1, normalize=normalize)
    hist2 = get_histvector_from_PIL(img2, normalize=normalize)

    return euclidean(hist1, hist2)


def imagepair_similarity(img1, img2):
    hist1 = get_histvector_from_PIL(img1)
    hist2 = get_histvector_from_PIL(img2)

    return 1 - cosine(hist1, hist2)



# Other metrics:
# https://towardsdatascience.com/measuring-similarity-in-two-images-using-python-b72233eb53c6
# https://github.com/andrewekhalel/sewar
