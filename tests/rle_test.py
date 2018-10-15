#!/usr/bin/python3
#
# Stephen Vondenstein, Matthew Buckley
# 10/14/2018
#
import os, subprocess
# Add parent to project root so we can import project files
import sys
sys.path.append('..')

from utils.parser import get_args
from utils.utility import create_dirs
from utils.utility import get_max_filename
from utils.utility import get_max_unused_filename
from utils.rle import prepare_submission

def compare_rle():
    config = get_args()
    # Ensure submission_path is created
    output_path = './rle_tests/'
    create_dirs([output_path])

    # Save path and filenames
    if config.rle:
        rle = prepare_submission(config.prediction_path, config.submission_path, 'iotest')
        test_path = rle
    else:
        input_path = './image_tests/'
        test_rle = get_max_filename(input_path, 'iotest', '.csv')
        test_path = os.path.join(input_path, test_rle)

    sort_rle = get_max_unused_filename(output_path, 'test', '.csv')
    sort_path = os.path.join(output_path, sort_rle)

    print('Comparing RLE values...')
    # Sort csv files
    subprocess.Popen("sort ./data/train.csv > ./rle_tests/source.csv", shell=True)
    subprocess.Popen("sort " + str(test_path) + " > " + str(sort_path), shell=True)
    # Compare RLEs for saved masks
    diff = subprocess.Popen("diff -as ./rle_tests/source.csv " + sort_path, shell=True,
                            stdout=subprocess.PIPE).stdout.readline().decode()
    print('Files ./image_tests/source.csv and ./image_tests/test.csv '
          'are not identical') if 'identical' not in diff else print(diff)


if __name__ == '__main__':
    # If executing from this folder, change the working directory to the parent to manipulate project files
    if os.path.dirname(os.path.abspath(__file__)) == os.getcwd():
        project_root = os.path.abspath('..')
        os.chdir(project_root)
    compare_rle()