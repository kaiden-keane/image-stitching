import cv2 as cv
import numpy as np
import math

class Vec2d():
    def __init__(self, x:int=0 , y:int=0):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"

def get_translation(matches: list[cv.DMatch]) -> tuple[int, int]:
    """
    input: a list of cv.DMatch
    returns a tuple, (x, y) for translation that needs to occure to overlay the image
    """
    translation = Vec2d()
    amnt = 0
    for match in matches:
        if amnt > 10:
            break
        translation.x += kp1[match.queryIdx].pt[0] - kp2[match.trainIdx].pt[0]
        translation.y += kp1[match.queryIdx].pt[1] - kp2[match.trainIdx].pt[1]
        amnt += 1

    translation.x /= amnt
    translation.y /= amnt
    
    return (int(math.ceil(translation.x)), int(math.ceil(translation.y)))


img1 = cv.imread('sample_images/left.png') # queryImage / base image
img2 = cv.imread('sample_images/right.png') # trainImage / translated image

orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

translation = get_translation(matches)

img1_offset = Vec2d()
img2_offset = Vec2d()
if translation[0] < 0:
    img2_offset.x = 0
    img1_offset.x = translation[0] * -1
else:
    img2_offset.x = translation[0]
    img1_offset.x = 0
if translation[1] < 0:
    img2_offset.y = 0
    img1_offset.y = translation[1] * -1
else:
    img2_offset.y = translation[1]
    img1_offset.y = 0

# get bounding box of both images after translation
blank_width = max(img1.shape[1], img2.shape[1] + translation[0]) - min(0, translation[0])
blank_height = max(img1.shape[0], img2.shape[0] + translation[1]) - min(0, translation[1])

blank_image_dimensions = Vec2d(blank_width, blank_height)

blank_image = np.zeros((blank_image_dimensions.y, blank_image_dimensions.x, 3), np.uint8)

blank_image[img2_offset.y:img2_offset.y + img2.shape[0], img2_offset.x:img2_offset.x + img2.shape[1]] = img2
blank_image[img1_offset.y:img1_offset.y + img1.shape[0], img1_offset.x:img1_offset.x + img1.shape[1]] = img1

cv.imshow('result', blank_image)
cv.waitKey(0)
cv.imwrite('output.png', blank_image)