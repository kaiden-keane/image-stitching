import cv2 as cv
import numpy as np
import math
import os


class Vec2d():
    """
    simple x, y vector
    """
    def __init__(self, x:int=0 , y:int=0):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"
    
    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"


class BaseImage():
    """
    contains the data pertaining to one image
    key points, features, size, etc
    """
    def __init__(self, filename): 
        if (os.path.exists(filename)):
            self.image = cv.imread(filename)
        else:
            raise FileNotFoundError
        self.offset = Vec2d()
        self.kp = []
        self.des = []
        

    def get_features(self, orb):
        self.kp, self.des = orb.detectAndCompute(self.image, None)

class ImageMatch():
    def __init__(self, img1, img2, top_match_count=None):
        self.img1 = img1
        self.img2 = img2
        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(img1.des, img2.des)
        # get the best x matches
        if top_match_count == None: match_count = len(matches)
        else: match_count = top_match_count
        
        top_matches = self._get_top_matches(matches, match_count)
        self.translation = self._get_translation(img1, img2, top_matches, match_count)
        self._calc_offset(img1, img2, self.translation)

    def _get_top_matches(self, matches, match_count):
        assert match_count <= len(matches)
        top_arr = []
        for i in range(match_count):
            lowest_val = matches[i]
            for j in range(1, len(matches)):
                if matches[j].distance < lowest_val.distance:
                    lowest_val = matches[j]
            top_arr.append(lowest_val)
        
        
        self.top_matches = top_arr

        self.top_matches = top_arr
        return top_arr


    def _get_translation(self, img1, img2, top_matches, match_count) -> Vec2d:
        """
        input: a list of cv.DMatch
        returns a tuple, (x, y) q2for translation that needs to occure to overlay the image
        """
        assert match_count > 0, "ImageMatcher.match_count <= 0"
        translation = Vec2d()
        amnt = 0
        for match in top_matches:
            if amnt > match_count:
                break
            translation.x += img1.kp[match.queryIdx].pt[0] - img2.kp[match.trainIdx].pt[0]
            translation.y += img1.kp[match.queryIdx].pt[1] - img2.kp[match.trainIdx].pt[1]
            amnt += 1

        translation.x = math.ceil(translation.x / amnt)
        translation.y = math.ceil(translation.y / amnt)

        return translation

    def _calc_offset(self, img1, img2, translation):
        if translation.x < 0:
            img2.offset.x = 0
            img1.offset.x = translation.x * -1
        else:
            img2.offset.x = translation.x
            img1.offset.x = 0
        if translation.y < 0:
            img2.offset.y = 0
            img1.offset.y = translation.y * -1
        else:
            img2.offset.y = translation.y
            img1.offset.y = 0


def stitch_images(image_match):
    img1 = image_match.img1
    img2 = image_match.img2
    blank_width = max(img1.image.shape[1], img2.image.shape[1] + image_match.translation.x) - min(0, image_match.translation.x)
    blank_height = max(img1.image.shape[0], img2.image.shape[0] + image_match.translation.y) - min(0, image_match.translation.y)
    blank_image_dimensions = Vec2d(blank_width, blank_height)
    blank_image = np.zeros((blank_image_dimensions.y, blank_image_dimensions.x, 3), np.uint8)


    blank_image[img2.offset.y: img2.offset.y + img2.image.shape[0], img2.offset.x: img2.offset.x + img2.image.shape[1]] = img2.image
    blank_image[img1.offset.y: img1.offset.y + img1.image.shape[0], img1.offset.x: img1.offset.x + img1.image.shape[1]] = img1.image

    return blank_image



if __name__ == "__main__":
    orb = cv.ORB_create()

    img1 = BaseImage(os.path.join("sample_images", "right.png")) # queryImage / base image
    img2 = BaseImage(os.path.join("sample_images", "left.png")) # trainImage / translated image

    img1.get_features(orb)
    img2.get_features(orb)

    image_match = ImageMatch(img1, img2)
    combined_img = stitch_images(image_match)

    # cv.imshow("output", combined_img)
    # cv.waitKey(0)

    cv.imwrite("output.png", combined_img)