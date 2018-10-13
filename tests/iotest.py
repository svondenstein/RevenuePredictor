#
# Stephen Vondenstein, Matthew Buckley
# 10/12/2018
#
from helpers.data_generator import DataGenerator
import cv2
import os
from utils.parser import get_args

def main():
    config = get_args()
    data_loader = DataGenerator(config)
    image, mask, name = data_loader.get_input()
    cv2.imwrite(os.path.join('./image_tests/', name.decode('utf-8')), 255 * image)

if __name__ == '__main__':
    main()