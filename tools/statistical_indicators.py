import pdb
import os
import argparse


def main(args):
    folder = args.folder
    with open(os.path.join(folder, 'log.txt'), 'r') as f:
        val_lines = [line for line in f if 'val' in line]
        for line in val_lines:
            print(line)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Statistical Indicators')
    parser.add_argument('--folder', type=str, help='Input folder')
    args = parser.parse_args()
    main(args)
