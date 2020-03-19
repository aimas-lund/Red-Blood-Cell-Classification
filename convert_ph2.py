from DataHandler import DataHandler

dh = DataHandler()
path = r'C:/Users/Aimas/Desktop/DTU/01-BSc/6_semester/01_Bachelor_Project/data/01-4-40x-6mbar-500fps/raw'
path_out = r'C:/Users/Aimas/Desktop/DTU/01-BSc/6_semester/01_Bachelor_Project/data/01-4-40x-6mbar-500fps/jpgs/'
dh.png2jpg(path, path_out)