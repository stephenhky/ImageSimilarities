
import os
from argparse import ArgumentParser
from operator import itemgetter

from PIL import Image
from tqdm import tqdm

from imgutils import imagepair_distance, imagepair_similarity


def get_argparser():
    argparser = ArgumentParser(description='Sorting image similarity.')
    argparser.add_argument('--euclidean', default=False, action='store_true', help='Use Euclidean distance. (Default: using cosine similarity)')
    argparser.add_argument('image1path', help='path of the reference image')
    argparser.add_argument('imagepaths', nargs='+', help='paths of images to compare')
    argparser.add_argument('--n', default=10, type=int, help='number of files shown. (Default: 10)')
    return argparser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    eudist = args.euclidean
    img1 = Image.open(args.image1path)
    imagepaths = args.imagepaths
    n = args.n

    resultdict = {}
    for imagepath in tqdm(imagepaths):
        img = Image.open(imagepath)
        if eudist:
            value = imagepair_distance(img1, img, normalize=True)
        else:
            value = imagepair_similarity(img1, img)
        resultdict[imagepath] = value

    for imgpath, value in sorted(resultdict.items(), key=itemgetter(1), reverse=eudist)[-n:]:
        basefilename = os.path.basename(imgpath)
        print('{}: {}'.format(basefilename, value))



