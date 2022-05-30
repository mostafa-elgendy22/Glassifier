import cv2
import numpy as np
from PIL import Image, ImageEnhance


# hinge feature parameters
p = 40                              # number of angle bins
bin_size = 360 // p
r = 25                              # leg length

def preprocess_image(image: np.ndarray, sharpness_factor = 10, bordersize = 3):
        image = Image.fromarray(image)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness_factor)

        (height, width) = (image.height * 2, image.width * 2)
        image = image.resize((width, height))

        image = np.asarray(image)
        image = cv2.copyMakeBorder(
            image,
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (3, 3), 0)

        (_, preprocessed_image) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return preprocessed_image


def get_contour_pixels(preprocessed_image: np.ndarray, th1 = 100000, th2 = 1000):
    imgh, imgw = np.shape(preprocessed_image)
    imgarea = imgh * imgw
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # contours_filt = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
    contours_filt = list(filter(lambda cnt: imgarea/th1 <= cv2.contourArea(cnt) <= imgarea/th2, contours))

    return contours_filt


def get_hinge_features(image_path = None, image = None):
    if image is None:
        image = cv2.imread(image_path)

    preprocessed_image = preprocess_image(image)
    contours = get_contour_pixels(preprocessed_image)

    hist = np.zeros((p, p))

    for cnt in contours:
        n_pixels = len(cnt)
        if n_pixels <= r:
            continue

        points = np.array([point[0] for point in cnt])
        (xs, ys) = (points[:, 0], points[:, 1])
        point_1s = np.array([cnt[(i + r) % n_pixels][0] for i in range(n_pixels)])
        point_2s = np.array([cnt[(i - r) % n_pixels][0] for i in range(n_pixels)])
        (x1s, y1s) = (point_1s[:, 0], point_1s[:, 1])
        (x2s, y2s) = (point_2s[:, 0], point_2s[:, 1])

        phi_1s = np.degrees(np.arctan2(y1s - ys, x1s - xs) + np.pi)
        phi_2s = np.degrees(np.arctan2(y2s - ys, x2s - xs) + np.pi)

        indices = np.where(phi_2s > phi_1s)[0]

        for i in indices:
            phi1 = int(phi_1s[i] // bin_size) % p
            phi2 = int(phi_2s[i] // bin_size) % p
            hist[phi1, phi2] += 1

    normalized_hist = hist / np.sum(hist)
    feature_vector = normalized_hist[np.triu_indices_from(normalized_hist, k=1)]
    return feature_vector
