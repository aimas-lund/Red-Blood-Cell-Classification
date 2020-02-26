import pickle as pkl
import numpy as np
from PIL import Image
import glob
import os


def pickle_images(images, path='./data.pkl', protocol=3):
    with open(path, 'wb') as file:
        pkl.dump(images, file, protocol=protocol)
    file.close()


def images_to_array_list(path, file_extension='png', flatten=False):
    """
    Loads all images of a specified file-type (png by default) in a directory to a list of Numpy arrays
    :param flatten: Optionally compresses the array into a single dimension
    :param path: Path to directory containing images
    :param file_extension: File extension to search for.
    :return: List of Numpy arrays
    """
    output = []

    # finds every filename in the specified directory and iterates over any file with the specified extension
    for filename in glob.glob(os.path.join(path, '*.' + file_extension)):
        x = np.asarray(Image.open(filename))

        if flatten:
            x = x.flatten()

        output.append(x)

    return output
