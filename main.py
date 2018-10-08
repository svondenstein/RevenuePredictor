#
# Stephen Vondenstein, Matthew Buckley
# 10/07/2018
#
import argparse
from helpers import get_data_paths_list
from helpers import save_model_params
from helpers import load_model_params
from model import DenseTiramisu

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="infer")
parser.add_argument("--train_data", default="train",
                    help="Directory for training images")
parser.add_argument("--val_data", default="train",
                    help="Directory for validation images")
parser.add_argument("--ckpt", default="checkpoints/model.ckpt",
                    help="Directory for storing model checkpoints")
parser.add_argument("--layers_per_block", default="4,5,7,10,12,15",
                    help="Number of layers in dense blocks")
parser.add_argument("--batch_size", default=4,
                    help="Batch size for use in training", type=int)
parser.add_argument("--epochs", default=12,
                    help="Number of epochs for training", type=int)
parser.add_argument("--num_threads", default=2,
                    help="Number of threads to use for data input pipeline", type=int)
parser.add_argument("--growth_k", default=16, help="Growth rate for Tiramisu", type=int)
parser.add_argument("--num_classes",   default=2, help="Number of classes", type=int)
parser.add_argument("--learning_rate", default=1e-4,
                    help="Learning rate for optimizer", type=float)
parser.add_argument("--infer_data", default="test")
parser.add_argument("--output_folder", default="predictions")
parser.add_argument("--prior_model", default="")



def main():
    FLAGS = parser.parse_args()

    if FLAGS.mode == 'train':
        save_model_params(FLAGS.growth_k, FLAGS.layers_per_block, FLAGS.num_classes, FLAGS.ckpt)
        layers_per_block = [int(x) for x in FLAGS.layers_per_block.split(",")]
        tiramisu = DenseTiramisu(FLAGS.growth_k, layers_per_block, FLAGS.num_classes)
        tiramisu.train(FLAGS.train_data, FLAGS.val_data, FLAGS.ckpt,
        FLAGS.batch_size, FLAGS.epochs, FLAGS.learning_rate, FLAGS.prior_model)
    elif FLAGS.mode == 'infer':
        growth_k, layers_per_block, num_classes = load_model_params(FLAGS.ckpt)
        layers_per_block = [int(x) for x in layers_per_block.split(",")]
        tiramisu = DenseTiramisu(growth_k, layers_per_block, num_classes)
        tiramisu.infer(FLAGS.infer_data, 1, FLAGS.ckpt, FLAGS.output_folder)


if __name__ == "__main__":
    main()
