from PIL import Image
import argparse
import sys
import os

if(__name__ == "__main__"):

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--directory", type = str, help = "The directory of the dataset")

    args = parser.parse_args()

    if(args.directory == None):
        sys.exit("directory is a required argument")

    def clean_dir(dir_name):
        for img in os.listdir(os.path.join(args.directory, dir_name)):
            width, height = Image.open(os.path.join(args.directory, "train", img)).size

            if(width < 64 or height < 64):
                os.remove(os.path.join(args.directory, dir_name, img))

    # training data
    clean_dir("train")
    # validation data
    clean_dir("valid")
