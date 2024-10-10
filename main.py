import cv2 as cv
import numpy as np
import math
import os

# HI Kaiden :)
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


class ImageMatcher():
    """
    contains instance of ORB and matches two images
    calculates translation and offset required to properly overlay images
    """
    def __init__(self) -> None:
        self.orb = cv.ORB_create()

    def get_matches(self, img1, img2, top_count=None):
        self.img1 = img1
        self.img2 = img2
        
        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(img1.des, img2.des)
        if top_count == None: self.match_count = len(matches)
        else: self.match_count = top_count

        # get the best x matches
        self.top_matches = self._get_top_features(matches)
        self._get_translation()
        self._get_offset()

    def _get_top_features(self, matches):
        assert self.match_count <= len(matches)
        top_arr = []
        for i in range(self.match_count):
            lowest_val = matches[i]
            for j in range(1, len(matches)):
                if matches[j].distance < lowest_val.distance:
                    lowest_val = matches[j]
            top_arr.append(lowest_val)
        
        self.top_matches = top_arr
        return top_arr

    def _get_translation(self) -> Vec2d:
        """
        input: a list of cv.DMatch
        returns a tuple, (x, y) q2for translation that needs to occure to overlay the image
        """
        assert self.match_count > 0, "ImageMatcher.match_count <= 0"
        translation = Vec2d()
        amnt = 0
        for match in self.top_matches:
            if amnt > self.match_count:
                break
            translation.x += self.img1.kp[match.queryIdx].pt[0] - self.img2.kp[match.trainIdx].pt[0]
            translation.y += self.img1.kp[match.queryIdx].pt[1] - self.img2.kp[match.trainIdx].pt[1]
            amnt += 1

        translation.x = math.ceil(translation.x / amnt)
        translation.y = math.ceil(translation.y / amnt)
        self.translation = translation
        return translation
    
    def _get_offset(self):
        if self.translation.x < 0:
            self.img2.offset.x = 0
            self.img1.offset.x = self.translation.x * -1
        else:
            self.img2.offset.x = self.translation.x
            self.img1.offset.x = 0
        if self.translation.y < 0:
            self.img2.offset.y = 0
            self.img1.offset.y = self.translation.y * -1
        else:
            self.img2.offset.y = self.translation.y
            self.img1.offset.y = 0


class Stitcher():
    """
    currently just overlays two images but will make it more convenient to match bulk images
    """
    def __init__(self, matcher) -> None:
        self.matcher = matcher
        img1 = matcher.img1.image
        img2 = matcher.img2.image
        blank_width = max(img1.shape[1], img2.shape[1] + matcher.translation.x) - min(0, matcher.translation.x)
        blank_height = max(img1.shape[0], img2.shape[0] + matcher.translation.y) - min(0, matcher.translation.y)
        blank_image_dimensions = Vec2d(blank_width, blank_height)
        self.blank_image = np.zeros((blank_image_dimensions.y, blank_image_dimensions.x, 3), np.uint8)
    
    def stitch(self):
        img1 = self.matcher.img1
        img2 = self.matcher.img2
        
        self.blank_image[img2.offset.y: img2.offset.y + img2.image.shape[0], img2.offset.x: img2.offset.x + img2.image.shape[1]] = img2.image
        self.blank_image[img1.offset.y: img1.offset.y + img1.image.shape[0], img1.offset.x: img1.offset.x + img1.image.shape[1]] = img1.image

    
    def show_output(self):
        cv.imshow("result", self.blank_image)
        cv.waitKey(0)
    
    def save_output(self):
        cv.imwrite("output.png", self.blank_image)


if __name__ == "__main__":
    matcher = ImageMatcher()

    img1 = BaseImage(os.path.join("sample_images", "img1.png")) # queryImage / base image
    img2 = BaseImage(os.path.join("sample_images", "img2.png")) # trainImage / translated image

    img1.get_features(matcher.orb)
    img2.get_features(matcher.orb)

    matcher.get_matches(img1, img2)
    
    stitcher = Stitcher(matcher)
    stitcher.stitch()
    # stitcher.show_output()
    stitcher.save_output()