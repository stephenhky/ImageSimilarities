
import numpy as np
from scipy.spatial.distance import euclidean, cosine
import cv2


# convert PIL image to OpenCV image: https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
def convert_PIL_to_opencv(pilimg):
    npimg = np.array(pilimg.convert('RGB'))
    cvimg = npimg[:, :, ::-1].copy()
    return cvimg


# Reference: https://www.geeksforgeeks.org/measure-similarity-between-images-using-python-opencv/
def histdistance(img1, img2, normalize=True):
    cvimg1 = convert_PIL_to_opencv(img1)
    cvimg2 = convert_PIL_to_opencv(img2)

    grayimg1 = cv2.cvtColor(cvimg1, cv2.COLOR_BGR2GRAY)
    grayimg2 = cv2.cvtColor(cvimg2, cv2.COLOR_BGR2GRAY)

    hist1 = cv2.calcHist([grayimg1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([grayimg2], [0], None, [256], [0, 256])

    # normalize
    if normalize:
        hist1 /= np.linalg.norm(hist1)
        hist2 /= np.linalg.norm(hist2)

    return euclidean(hist1, hist2)
