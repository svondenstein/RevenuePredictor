#
# Stephen Vondenstein, Matthew Buckley
# 10/08/2018
#
import os


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

def get_max_filename(directory, prefix, extension):
    i = 1
    while os.path.exists(os.path.join(directory, '%s-%s%s' % (prefix, i, extension))):
        i += 1
    file_name = '%s%s-%s%s' % (directory, prefix, i, extension)

    return file_name

def get_max_unused_filename(directory, prefix, extension):
    i = 1
    while os.path.exists(os.path.join(directory, '%s-%s%s' % (prefix, i, extension))):
        i += 1
    i + 1
    file_name = '%s%s-%s%s' % (directory, prefix, i, extension)

    return file_name