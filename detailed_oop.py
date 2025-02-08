import cv2 as cv
import numpy as np
import os
import time

#----------------------------------------------------------
# based on stitching_detailed found below:
# https://docs.opencv.org/4.x/d8/d19/tutorial_stitcher.html
#----------------------------------------------------------


images_dir = "sample_images"
output_dir = "result_images"
output_name = "detailed_oop.png"

def get_image_names():
    # get all file names from sample images directory
    fileNames = os.listdir(images_dir)
    
    # files <= only valid files + full path
    for name in fileNames:
        if not name.endswith(".jpg") and not name.split(".")[0].isdigit():
            fileNames.remove(name)
    fileNames.sort(key = lambda x: int(x.split(".")[0])) # sort based on numerical order

    fileNames = [ os.path.join(images_dir, x) for x in fileNames]

    return fileNames


class Stitcher():
    match_confidence = 0.3 # match confidence for orb feature matching
    work_megapix= 0.6 # == registrationResol
    seam_megapix= 0.1 # == seamEstimationResol
    
    # we likely dont want full resolution in the end so make compose_megapix different than original
    compose_megapix = -1 # -1 = original resolution
    conf_thresh = 0.3 # threshhold for two images are from the same panormama confidence is 0

    warp_type = "affine"
    result_name = output_name
    seam_work_aspect = 0.5
    
    def __init__(self):
        self.feature_detector = cv.ORB.create()
        self.matcher = cv.detail.AffineBestOf2NearestMatcher(False, True, self.match_confidence)

    def __registerImages():
        pass

    def __composeImages():
        pass


    def stitch(self, img_names):

        seam_work_aspect = 1
        full_img_sizes = []
        features = []
        images = []
        is_work_scale_set = False
        is_seam_scale_set = False
        is_compose_scale_set = False

        # load and convert images to proper scale
        start = time.time()
        for i, name in enumerate(img_names):
            full_img = cv.imread(name)
            if full_img is None:
                print("Cannot read image ", name)
            else:
                full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
                
                if is_work_scale_set is False:
                    work_scale = min(1.0, np.sqrt(self.work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                    is_work_scale_set = True
                img = cv.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
                
                if is_seam_scale_set is False:
                    seam_scale = min(1.0, np.sqrt(self.seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                    seam_work_aspect = seam_scale / work_scale
                    is_seam_scale_set = True

                # compute features
                img_feat = cv.detail.computeImageFeatures2(self.feature_detector, img)
                features.append(img_feat)
                img = cv.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
                images.append(img)

        end = time.time()
        print(f"time spent loading, rescaling, getting features for images: {end-start}")
        
        # ensure images are here are only images actaully from the same scan
        start = time.time()
        
        p = self.matcher.apply2(features)
        self.matcher.collectGarbage()

        indices = cv.detail.leaveBiggestComponent(features, p, self.conf_thresh)
        img_subset = []
        img_names_subset = []
        full_img_sizes_subset = []
        for i in range(len(indices)):
            img_names_subset.append(img_names[indices[i]])
            img_subset.append(images[indices[i]])
            full_img_sizes_subset.append(full_img_sizes[indices[i]])
        images = img_subset
        img_names = img_names_subset
        full_img_sizes = full_img_sizes_subset
        num_images = len(img_names)
        if num_images < 2:
            print("Need more images")
            exit()
        
        end = time.time()
        print(f"time spent getting correct image subset: {end-start}")


        start = time.time()
        estimator = cv.detail.AffineBasedEstimator()
        b, cameras = estimator.apply(features, p, None)
        if not b:
            print("Homography estimation failed.")
            exit()
        for cam in cameras:
            cam.R = cam.R.astype(np.float32)

        adjuster = cv.detail.BundleAdjusterAffinePartial()
        adjuster.setConfThresh(self.conf_thresh)
        adjuster.setRefinementMask(np.ones((3, 3), np.uint8)) # == ba_refine_mask == xxxxx
        b, cameras = adjuster.apply(features, p, cameras)
        if not b:
            print("Camera parameters adjusting failed.")
            exit()
        
        focals = []
        for cam in cameras:
            focals.append(cam.focal)
        focals.sort()
        if len(focals) % 2 == 1:
            warped_image_scale = focals[len(focals) // 2]
        else:
            warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
        
        end = time.time()
        print(f"time spent on estimator: {end-start}")


        start = time.time()
        corners = []
        masks_warped = []
        images_warped = []
        sizes = []
        masks = []
        for i in range(0, num_images):
            um = cv.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8))
            masks.append(um)

        warper = cv.PyRotationWarper(self.warp_type, warped_image_scale * seam_work_aspect)  # warper could be nullptr?
        for idx in range(0, num_images):
            K = cameras[idx].K().astype(np.float32)
            swa = seam_work_aspect
            K[0, 0] *= swa
            K[0, 2] *= swa
            K[1, 1] *= swa
            K[1, 2] *= swa
            corner, image_wp = warper.warp(images[idx], K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
            corners.append(corner)
            sizes.append((image_wp.shape[1], image_wp.shape[0]))
            images_warped.append(image_wp)
            p, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
            masks_warped.append(mask_wp.get())
        images_warped_f = []
        for img in images_warped:
            imgf = img.astype(np.float32)
            images_warped_f.append(imgf)
        
        end = time.time()
        print(f"time spent warping images: {end-start}")


        start = time.time()
        compensator = cv.detail.ExposureCompensator.createDefault(cv.detail.ExposureCompensator_NO)
        compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

        # I believe we can get rid of this?
        end = time.time()
        print(f"time spent on exposure compensator: {end-start}")


        start = time.time()
        seam_finder = cv.detail.GraphCutSeamFinder('COST_COLOR')
        masks_warped = seam_finder.find(images_warped_f, corners, masks_warped)
        
        end = time.time()
        print(f"time spent on seam finder: {end-start}")
        

        start = time.time()

        compose_scale = 1
        corners = []
        sizes = []
        blender = None

        for idx, name in enumerate(img_names):
            full_img = cv.imread(name)
            
            # on first iteration set compose_scale
            if not is_compose_scale_set:
                if self.compose_megapix > 0:
                    compose_scale = min(1.0, np.sqrt(self.compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                is_compose_scale_set = True
                compose_work_aspect = compose_scale / work_scale
                warped_image_scale *= compose_work_aspect
                warper = cv.PyRotationWarper(self.warp_type, warped_image_scale)
                for i in range(0, len(img_names)):
                    cameras[i].focal *= compose_work_aspect
                    cameras[i].ppx *= compose_work_aspect
                    cameras[i].ppy *= compose_work_aspect
                    sz = (int(round(full_img_sizes[i][0] * compose_scale)),
                        int(round(full_img_sizes[i][1] * compose_scale)))
                    K = cameras[i].K().astype(np.float32)
                    roi = warper.warpRoi(sz, K, cameras[i].R)
                    corners.append(roi[0:2])
                    sizes.append(roi[2:4])

            if abs(compose_scale - 1) > 1e-1:
                img = cv.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale,
                                interpolation=cv.INTER_LINEAR_EXACT)
            else:
                img = full_img

            K = cameras[idx].K().astype(np.float32)

            corner, image_warped = warper.warp(img, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
            
            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
            compensator.apply(idx, corners[idx], image_warped, mask_warped)
            image_warped_s = image_warped.astype(np.int16)
            dilated_mask = cv.dilate(masks_warped[idx], None)
            seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT)
            mask_warped = cv.bitwise_and(seam_mask, mask_warped)
            
            # if blender is not created, make it
            if blender is None:
                blender = cv.detail.Blender.createDefault(cv.detail.Blender_NO)
                dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
                blender.prepare(dst_sz)
            
            # add the image to be blended
            blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx])
        
        end = time.time()
        print(f"time spent on stitching: {end-start}")


        # save the image
        start = time.time()
        # blend and return the final pano
        result = None
        result_mask = None
        result, result_mask = blender.blend(result, result_mask)

        end = time.time()
        print(f"time spent on blending: {end-start}")

        start = time.time()
        cv.imwrite(os.path.join(output_dir, self.result_name), result)
        
        end = time.time()
        print(f"time spent saving image: {end-start}")


if __name__ == "__main__":
    stitcher = Stitcher()
    names = get_image_names()
    start = time.time()
    stitcher.stitch(names[:15])
    end = time.time()
    print(f"program took {end - start} seconds")

