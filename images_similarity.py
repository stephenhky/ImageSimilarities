
import os
from argparse import ArgumentParser

from PIL import Image

from imgutils import imagepair_distance, imagepair_similarity


def get_argparser():
    argparser = ArgumentParser(description='Sorting image similarity.')
    argparser.add_argument('--euclidean', default=False, action='store_true', help='Use Euclidean distance. (Default: using cosine similarity)')
    argparser.add_argument('image1path', help='path of image1')
    argparser.add_argument('image2path', help='path of image2')
    return argparser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    eudist = args.euclidean
    img1 = Image.open(args.image1path)
    img2 = Image.open(args.image2path)

    if eudist:
        value = imagepair_distance(img1, img2, normalize=True)
    else:
        value = imagepair_similarity(img1, img2)

    basefilename1 = os.path.basename(args.image1path)
    basefilename2 = os.path.basename(args.image2path)
    print(basefilename1)
    print(basefilename2)
    print('similarity: {}'.format(value))



