import cv2 as cv
import numpy as np
import math
import os


images_directory = "sample_images"
output_dir = "result_images"
output_name = "manual.png"
filetype = "jpg"

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
    def __init__(self, data): 
        self.image = data
        self.offset = Vec2d()
        self.translation = Vec2d()
        self.kp = []
        self.des = []
        
    @classmethod
    def load_from_file(cls, filename):
        if (os.path.exists(filename)):
            data = cv.imread(filename)
            return cls(data)
        else:
            raise FileNotFoundError 

    @classmethod
    def load_from_array(cls, arr):
        return cls(arr)
        

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
        self.img2.translation = self._get_translation(img1, img2, top_matches, match_count)

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


def stitch_images(images):
    # find the size
    max_translation_i = Vec2d()
    min_translation_i = Vec2d()
    
    # all relative to first element
    for i in range(1, len(images)):
        # make relative translations absolution
        x_translation = images[i].translation.x + images[i-1].translation.x
        images[i].translation.x = x_translation
        y_translation = images[i].translation.y + images[i-1].translation.y
        images[i].translation.y = y_translation
        if x_translation > images[max_translation_i.x].translation.x:
            max_translation_i.x = i
        elif x_translation < images[min_translation_i.x].translation.x:
            min_translation_i.x = i
        if y_translation > images[max_translation_i.y].translation.y:
            max_translation_i.y = i
        elif y_translation < images[min_translation_i.y].translation.y:
            min_translation_i.y = i
        print(x_translation, y_translation)

    width = max(images[0].image.shape[1], images[max_translation_i.x].image.shape[1] + images[max_translation_i.x].translation.x) - min(0, images[min_translation_i.x].translation.x)
    height = max(images[0].image.shape[0], images[max_translation_i.y].image.shape[0] + images[max_translation_i.y].translation.y) - min(0, images[min_translation_i.y].translation.y)
    
    blank_image_dimensions = Vec2d(width, height)
    print(f"\nblank dimensions: {blank_image_dimensions}\n")
    blank_image = np.zeros((blank_image_dimensions.y, blank_image_dimensions.x, 3), np.uint8)

    min_trans_x = images[min_translation_i.x].translation.x
    min_trans_y = images[min_translation_i.y].translation.y
    print("min translation: ", end="")
    print(min_trans_x, min_trans_y)
    i = 0
    for img in images:
        # print(img2.translation)
        if min_trans_x < 0: img.offset.x = img.translation.x - min_trans_x
        else: img.offset.x = img.translation.x
        
        if min_trans_y < 0: img.offset.y = img.translation.y - min_trans_y
        else: img.offset.y = img.translation.y
        
        print(img.offset)

        blank_image[img.offset.y: img.offset.y + img.image.shape[0], img.offset.x: img.offset.x + img.image.shape[1]] = img.image

    return blank_image



if __name__ == "__main__":
    orb = cv.ORB_create()
    files = sorted(
        [f for f in os.listdir(images_directory) if f.endswith('.' + filetype)])
    files = files[:10]

    images = []
    previous = BaseImage.load_from_file(os.path.join(images_directory, files[0]))
    previous.get_features(orb)
    images.append(previous)

    # assume image i+1 overlaps i
    for i in range(1, len(files)-1):
        img2 = BaseImage.load_from_file(os.path.join(images_directory, files[i]))
        img2.get_features(orb)
        match = ImageMatch(previous, img2, 10)
        images.append(img2)
        previous = img2

        
    stitched_image = stitch_images(images)
    resized = cv.resize(stitched_image, (0, 0), fx = 0.1, fy = 0.1)
    cv.imwrite(os.path.join(output_dir, output_name), resized)