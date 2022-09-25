#! python

# Example:
# python train.py -p "Caltec Database -faces" -i .jpg -t .txt -c builtin.json -m model.color.npz

from eigenface import *
import argparse
log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


def train(path, imgext, model):
    Emask = EigenFaceUtils()
    Emask.train(path, imgext, model)
    faces(model, Emask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--imgext", default=".jpg")
    args = parser.parse_args()
    train("./Train_DataSet", args.imgext, "Emodel3.npz")
