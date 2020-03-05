# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


def fetch_data(convert_gray=False, show=False):
    """
    Fetches data from the data directories TODO: Add labels
    :param convert_gray: boolean type to specify if images should be grayscale (normal is BRG blue-red-green).
    :param show: show the first sample of every category.
    :return: a list containing all imported images.
    """

    DATA_DIR = "./data"
    CATEGORIES = ["None", "Fetus", "Adult"]
    # TEST_CATEGORY = ["test"]
    
    images = []
    
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category) # define path to the different category data samples
        for img in os.listdir(path):
            if convert_gray:
                images.append(cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE))
            else:
                images.append(cv2.imread(os.path.join(path, img)))
                
        if show:
            if convert_gray:
                plt.imshow(images[0], cmap='gray')
            else:
                plt.imshow(images[0])
            plt.show()
            
        return images