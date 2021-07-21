#!/usr/bin/env python


import argparse
import copy
import signal
import sys
import time

import numpy as np

from grasp_samplers.grasp_model import GraspModel

MODEL_URL = "https://www.dropbox.com/s/fta8zebyzfrt3fw/checkpoint.pth.20?dl=0"
N_SAMPLES = 100
PATCH_SIZE = 100
model_name="model.pth"
url=MODEL_URL

from IPython import embed
import cv2
#cv2 = try_cv2_import()


def main():
    """
    This is the main function for running the grasping demo.
    """
    # initialize deep grasping network
    grasp_model = GraspModel(
            model_name=model_name, url=url, nsamples=N_SAMPLES, patchsize=PATCH_SIZE
        )


    for i in range(3):
        # read image from file
        img_original = cv2.imread('data/images/food/food_'+ (i+1) +'.jpg')

        # convert image from RGB to BGR
        img = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
        # # crop img into 240x440 pixels
        # img = img_1[dims[0][0] : dims[0][1], dims[1][0] : dims[1][1]]

        # selected_grasp = [183, 221, -1.5707963267948966, 1.0422693]
        selected_grasp = list(self.grasp_model.predict(img))

        # save pose estimation result
        cv2.imwrite('data/images/result/food_'+ (i+1) +'.jpg', grasp_model._disp_I)


if __name__ == "__main__":
    main()
