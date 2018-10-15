#
# Stephen Vondenstein, Matthew Buckley
# 10/08/2018
#
import os

# TODO: generalize to create from model param file
# Ensure model parameters 
def generate_params(config):
    params = {
        'learning_rate': config.learning_rate,
        ''


    }


def create_dirs(dirs):
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Error creating directories: {0}".format(err))
        exit(-1)

class AverageMeter:
    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg
