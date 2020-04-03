import os
import shutil
import random
import glob

def sample_images(source_path, dest_path, samples=100):

    source_path = source_path + "/*.png"
    to_be_moved = random.sample(glob.glob(source_path), samples)

    for f in enumerate(to_be_moved, 1):
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        shutil.copy(f[1], dest_path)
