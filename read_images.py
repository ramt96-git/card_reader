import os
import re
import cv2
import numpy as np
import pytesseract as pt
from random import randint

# Path to input image
img_dir = 'input/'
# path to save output images and detected text
op_dir = 'output/'
# Set as True to write intermediate detected edges and contours
debug_mode = True


def find_edges(image):
    # Convert RGB Image to GRAY, reduce Noise and find edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 20, 20, 20)
    edges = cv2.Canny(gray, threshold1=1300, threshold2=1800, apertureSize=7, L2gradient=True)

    return edges


def find_contours(edges):
    # Find Contours on the edges detected
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Convert identified contours to a convex blob
    hulls = [cv2.convexHull(x) for x in contours]
    # Take top 150 blobs sorted by area
    hulls = sorted(hulls, key=cv2.contourArea, reverse=True)[:150]
    # Combine overlapping blobs into single blob
    overlap = np.zeros((edges.shape[0], edges.shape[1]))
    for cnt in hulls:
        temp = np.zeros((edges.shape[0], edges.shape[1]))
        cv2.fillConvexPoly(temp, cnt, 255)
        overlap = np.bitwise_or(overlap.astype(np.int32), temp.astype(np.int32))

    overlap[overlap == -1] = 255
    # Convert the blobs into contours
    corrected_contours, _ = cv2.findContours(overlap.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Convert identified contours into convex blob or hull
    corrected_hulls = [cv2.convexHull(x) for x in corrected_contours if cv2.contourArea(x) >= 0.02 * edges.shape[0] * edges.shape[1]]

    return corrected_hulls


def find_min_area_rect(contours):
    # Identify nearest fitting rectangle that encapsulates given contour
    rect_contour = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rect_contour.append(box)
    return rect_contour


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def warp_rotate(contours, image, image_name):

    cropped_image_names = []
    for i, contour in enumerate(contours):
        # Four Point Transformation to convert coordinates into proper rectangle
        warped = four_point_transform(image, contour)
        # Identify correct orientation using Pytesseract
        angle = int(re.search('(?<=Rotate: )\d+', pt.image_to_osd(warped)).group(0))
        if angle == 90:
            rotated = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(warped, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotated = warped
        # Write rotated image to disk
        crop_img_name = image_name.replace('.', '_{}.'.format(i))
        cv2.imwrite(op_dir + crop_img_name, rotated)
        cropped_image_names.append(crop_img_name)

    return cropped_image_names


def run_tesseract(img_names):

    for c_img_name in img_names:
        # Run OCR on image
        text = pt.image_to_string(cv2.imread(op_dir + c_img_name))
        # Write detected Text to text file
        with open(op_dir + c_img_name.replace('.jpg', '.txt'), 'w+') as f:
            f.write(text)


if __name__ == '__main__':

    for img_name in [x for x in os.listdir(img_dir) if x.split('.')[-1].lower() == 'jpg']:

        print('Reading Image: {}'.format(img_name))
        img = cv2.imread(img_dir + img_name)
        print('Fining edges..')
        edge_image = find_edges(img)
        print('Finding contours..')
        img_contours = find_contours(edge_image)
        print('Approximating to nearest rectangle..')
        rect_contours = find_min_area_rect(img_contours)
        print('Warping and rotating to correct orientation..')
        card_image_names = warp_rotate(rect_contours, img.copy(), img_name)
        print('Successfully saved corrected card images!')
        print('Running OCR..')
        run_tesseract(card_image_names)
        print('Saved Text from Image Successfully!')

        if debug_mode:
            cv2.imwrite(op_dir + 'e_' + img_name, edge_image)
            for cnt in rect_contours:
                cv2.drawContours(img, [cnt], -1, (randint(0, 255), randint(0, 255), randint(0, 255)), 5)
            cv2.imwrite(op_dir + 'c_' + img_name, img)
