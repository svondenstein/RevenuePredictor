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

def compare_rle(source_path, test_path, output_path):
    # Ensure submission_path is created
    create_dirs([output_path])
    # Get paths
    test_rle = get_max_filename(source_path, 'test', '.csv')
    test_path = os.path.join(test_path, test_rle)
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
    # Get args and pass them to compare_rle
    config = get_args()
    compare_rle('./image_tests/', config.submission_path, './rle_tests')