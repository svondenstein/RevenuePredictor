#
# Stephen Vondenstein
# 10/07/2018
#
import os

def get_data_paths_list(image_folder, mask_folder):
    """Returns lists of paths to each image and mask."""

    image_paths = [os.path.join(image_folder, x) for x in os.listdir(
        image_folder) if x.endswith(".png")]
    mask_paths = [os.path.join(mask_folder, os.path.basename(x))
                  for x in image_paths]

    return image_paths, mask_paths

def save_model_params(growth_k, layers_per_block, num_classes, ckpt_path):
    param_path = ckpt_path.rsplit('/', 1)[0]
    param_path = param_path + '/model.params'
    print("Model parameters saved to: " + param_path)
    file = open(param_path, "w")
    data = [str(growth_k) + "\n", layers_per_block + "\n", str(num_classes) + "\n"]
    file.writelines(data)
    file.close()

def load_model_params(ckpt_path):
    param_path = ckpt_path.rsplit('/', 1)[0]
    param_path = param_path + '/model.params'
    print("Reading model paramaters from: " + param_path)
    file = open(param_path, "r")
    growth_k = int(file.readline())
    layers_per_block = file.readline()
    num_classes = int(file.readline())
    file.close()
    return growth_k, layers_per_block, num_classes
