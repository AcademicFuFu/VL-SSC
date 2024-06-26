import pdb
import os
import argparse


def main(args):
    folder = args.folder
    d = 3
    with open(os.path.join(folder, 'log.txt'), 'r') as f:
        val_lines = [line for line in f if 'val' in line]
        for epoch, line in enumerate(val_lines):
            for item in line.split(','):
                if 'mIoU' in item:
                    mIou = float(item.split()[-1])
                    print('epoch {}, mIou: {}'.format(str(epoch).zfill(2), str(mIou).zfill(5)), end='   ')
            if epoch % d == 0:
                print()
    files = os.listdir(folder)
    print(files)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Statistical Indicators')
    parser.add_argument('--folder', type=str, help='Input folder')
    args = parser.parse_args()
    main(args)
