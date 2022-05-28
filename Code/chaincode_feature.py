import cv2
import numpy as np
from collections import defaultdict

def chaincodes_from_contours(contours):
    # dx, dy -> direction
    dir = [['0', '1', '2'],
            ['7', '9', '3'],
            ['6', '5', '4']]

    conts = ['']*len(contours)
    for idx in range(len(contours)):
        for i in range(len(contours[idx])-1):
            dx, dy = (contours[idx][i+1] - contours[idx][i])[0]
            dx += 1
            dy += 1
            conts[idx] += dir[dy][dx]
    return conts


def probalize(bad_pdf):
    try:
        a = {key: value/sum(bad_pdf.values()) for key, value in bad_pdf.items()}
        return defaultdict(lambda: 0, a)
    except:
        a = {}
        return defaultdict(lambda: 0, a) # empty dict, avoid div by zero


def gen_labels(labels, LENGTH = 3, curr = ''):
    if (len(curr)):
        labels.append(curr)

    if len(curr) < LENGTH:
        for i in range(0, 8):
            gen_labels(labels, LENGTH, curr + str(i))
    return


def pdf_from_chaincodes(chaincodes, length):
    pdf = [defaultdict(lambda: 0, {}) for _ in range(length+1)]

    # calc freq of all substrings with length
    for leng in range(1, length+1):
        for cnt in chaincodes:
            for i in range(len(cnt) - leng):
                pdf[leng][cnt[i: i+leng]] += 1
    return pdf


def chaincodes_from_images(image, th1 = 100000, th2 = 1000):
    chaincodes = [[]]
    
    imgh, imgw = len(image), len(image[0])
    imgarea = imgh * imgw
    
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_OTSU)
    
    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    contours_filt = list(filter(lambda cnt: imgarea/th1 <= cv2.contourArea(cnt) <= imgarea/th2, contours))
    
    chaincodes[0] = chaincodes_from_contours(contours_filt)
    
    return chaincodes


def get_chaincode_features(image_path = None, image = None, th1 = 100000, th2 = 1000, MAX_CHAINCODE_LENGTH = 3):
    if image is None:
        image = cv2.imread(image_path)
    chaincodes = chaincodes_from_images(image, th1=th1, th2=th2)[0]
    pdfs = pdf_from_chaincodes(chaincodes=chaincodes, length=MAX_CHAINCODE_LENGTH)
    pdfs_n = [probalize(pdf) for pdf in pdfs]
    
    # generate all labels to fix order of features
    labels = []
    gen_labels(labels, MAX_CHAINCODE_LENGTH)

    # fix order!
    pdfs_combined = {label: 0 for label in labels}

    for pdf in pdfs_n:
        pdfs_combined.update(pdf)
    
    feature_vector = np.asarray([pdfs_combined[label] for label in labels])
    return feature_vector