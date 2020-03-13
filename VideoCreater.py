import cv2
import os
import sys
import time
from DataHandler import DataHandler

def extract(filename):
    return int(filename.split('_')[1].split('.')[0])

DATADIR = "C:/Users/Aimas/Desktop/DTU/01-BSc/6_semester/01_Bachelor_Project/data/01-4-40x-6mbar-500fps/"
handler = DataHandler(data_dir=DATADIR, categories=["raw"])

size = handler.detect_image_res()
path = os.path.join(DATADIR, "raw")  # define path to the different category data samples
listdir = os.listdir(path)

num_files = len(listdir)
num_file = 0
out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'mp4v'), 30, (size[1], size[0]))

for img in sorted(listdir, key=extract):
    image = cv2.imread(os.path.join(path, img))
    out.write(image)
    num_file += 1
    sys.stdout.write("\r{} of {} files loaded".format(num_file, num_files))
    sys.stdout.flush()

out.release()
