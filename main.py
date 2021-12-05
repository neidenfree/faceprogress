import os

import cv2
import dlib
import numpy as np
from imutils.face_utils import FaceAligner, rect_to_bb
import imutils

# cv2.getRotationMatrix2D([0, 1], 3402.2, 0.2)

#
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('venv/Include/shape_68.dat')
fa = FaceAligner(predictor, desiredFaceHeight=1600, desiredFaceWidth=768)

directory = 'photos/'
out_directory = 'output/'
images = os.listdir(directory)

full = len(images)

for i, image_file_name in enumerate(images):

    try:
        # load the input image, resize it, and convert it to grayscale
        image = cv2.imread(directory + image_file_name)
        image = imutils.resize(image, height=2400, width=1200)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 2)
        rect = rects[0]
        # rect = [(rect[0][0], rect[0][1]), (rect[1][0], rect[1][1])]

        # (x, y, w, h) = rect_to_bb(rect)
        # h += 200
        # faceOrig = imutils.resize(image[y:y + h, x:x + w], width=2048)
        faceAligned = fa.align(image, gray, rect)


        # display the output images

        # cv2.imshow("Original", faceOrig)
        # cv2.imshow("Aligned", faceAligned)


            # cv2.waitKey(0)

        cv2.imwrite(out_directory + image_file_name, faceAligned)

        print(f"{i} of {full} images is done")
    except:
        print(f"Some troubles with {i}-th image: {directory}/{image_file_name}")




# region Code below is a working code which get us a face mask with coordinates.


# def shape_to_np(shape, dtype="int"):
#     n = 68
#     coords = np.zeros((n, 2), dtype=dtype)
#     for i in range(0, n):
#         coords[i] = (shape.part(i).x, shape.part(i).y)
#     return coords
#
#
# def image_resize(image, width=100, height=None, inter=cv2.INTER_AREA):
#     # initialize the dimensions of the image to be resized and
#     # grab the image size
#     dim = None
#     (h, w) = image.shape[:2]
#
#     # if both the width and height are None, then return the
#     # original image
#     if width is None and height is None:
#         return image
#
#     # check to see if the width is None
#     if width is None:
#         # calculate the ratio of the height and construct the
#         # dimensions
#         r = height / float(h)
#         dim = (int(w * r), height)
#
#     # otherwise, the height is None
#     else:
#         # calculate the ratio of the width and construct the
#         # dimensions
#         r = width / float(w)
#         dim = (width, int(h * r))
#
#     # resize the image
#     resized = cv2.resize(image, dim, interpolation=inter)
#
#     # return the resized image
#     return resized
#
#
# def rotate(image, angle, center=None, scale=1.0):
#     (h, w) = image.shape[:2]
#
#     if center is None:
#         center = (w / 2, h / 2)
#
#     # Perform the rotation
#     M = cv2.getRotationMatrix2D(center, angle, scale)
#     rotated = cv2.warpAffine(image, M, (w, h))
#
#     return rotated
#
#
# directory = 'photos/'
# images = os.listdir(directory)
#
# img = image_resize(cv2.imread(directory + images[1]), width=800)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
# detector = dlib.get_frontal_face_detector()
# rects = detector(gray, 1)  # rects contains all the faces detected
# predictor = dlib.shape_predictor('venv/Include/shape_68.dat')
# for (i, rect) in enumerate(rects):
#     shape = predictor(gray, rect)
#
#     print(shape)
#     shape = shape_to_np(shape)
#     for (x, y) in shape:
#         cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
#
# # img.imshow()
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# endregion
