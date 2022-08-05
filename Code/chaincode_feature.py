import cv2
import numpy as np
from collections import defaultdict
from PIL import Image, ImageEnhance


def preprocess_image(image: np.ndarray, sharpness_factor = 10, bordersize = 3):
    image = Image.fromarray(image)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness_factor)
    image = np.asarray(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    (_, image) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel3)
    image = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel5)

    return image


def chaincodes_from_contours(contours):
    # dx, dy -> direction
    dir = [['0', '1', '2'],
            ['7', '9', '3'],
            ['6', '5', '4']]

    codes = ['']*len(contours)
    for idx in range(len(contours)):
        for i in range(len(contours[idx])-1):
            dx, dy = (contours[idx][i+1] - contours[idx][i])[0]
            dx += 1
            dy += 1
            codes[idx] += dir[dy][dx]
    return codes


# normalize pdfs
def probalize(bad_pdf):
    try:
        a = {key: value/sum(bad_pdf.values()) for key, value in bad_pdf.items()}
        return defaultdict(lambda: 0, a)
    except:
        a = {}
        return defaultdict(lambda: 0, a) # empty dict, avoid div by zero


# recursively get labels
# later used to assert all dictionaries use same order!
def gen_labels(labels, LENGTH = 3, curr = ''):
    if (len(curr)):
        labels.append(curr)

    if len(curr) < LENGTH:
        for i in range(0, 8):
            gen_labels(labels, LENGTH, curr + str(i))
    return


# pdfs[LENGTH][CODE] -> histogram
# pdfs[1]['5']
# pdfs[2]['37'] 
def pdfs_from_chaincodes(chaincodes, length):
    pdfs = [defaultdict(lambda: 0, {}) for _ in range(length+1)]

    # calc freq of all substrings with length
    for leng in range(1, length+1):
        for cnt in chaincodes:
            for i in range(len(cnt) - leng):
                pdfs[leng][cnt[i: i+leng]] += 1
    return pdfs


def chaincodes_from_image(image, th1 = 100000, th2 = 1000):    
    imgh, imgw = len(image), len(image[0])
    imgarea = imgh * imgw
    
    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, _ = cv2.findContours(image=image, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    
    # contours_filt = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
    contours_filt = list(filter(lambda cnt: imgarea/th1 <= cv2.contourArea(cnt) <= imgarea/th2, contours))
    
    return chaincodes_from_contours(contours_filt)


def get_chaincode_features(image_path = None, th1 = 100000, th2 = 1000, MAX_CHAINCODE_LENGTH = 3):
    image = cv2.imread(image_path)

    pre_processed_image = preprocess_image(image)
    chaincodes = chaincodes_from_image(pre_processed_image, th1=th1, th2=th2)[0]
    pdfs = pdfs_from_chaincodes(chaincodes=chaincodes, length=MAX_CHAINCODE_LENGTH)
    pdfs_n = [probalize(pdf) for pdf in pdfs]
    
    # generate all labels to fix order of features
    labels = []
    gen_labels(labels, MAX_CHAINCODE_LENGTH)

    pdfs_combined = {label: 0 for label in labels}
    for pdf in pdfs_n:
        pdfs_combined.update(pdf)
    
    feature_vector = np.asarray([pdfs_combined[label] for label in labels])
    return feature_vector