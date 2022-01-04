
from argparse import ArgumentParser
from PIL import Image
import os
from functools import lru_cache
from operator import itemgetter

from scipy.spatial.distance import cosine
from tqdm import tqdm

from imgutils import get_histvector_from_PIL


def get_argparser():
    argparser = ArgumentParser(description='Sorting image similarity.')
    argparser.add_argument('imagepaths', nargs='+', help='paths of images to compare')
    argparser.add_argument('--n', default=10, type=int, help='number of files shown. (Default: 10)')
    argparser.add_argument('--cachedsize', default=1000, type=int, help='cached size (Default: 1000)')
    return argparser


def get_cacched_histvecctor(imgpath):
    pilimg = Image.open(imgpath)
    return get_histvector_from_PIL(pilimg)


if __name__ == '__main__':
    args = get_argparser().parse_args()
    imagepaths = args.imagepaths
    n = args.n
    cachedsize = args.cachedsize

    get_cacched_histvecctor = lru_cache(cachedsize)(get_cacched_histvecctor)

    while True:
        path = input('Path> ')
        if path is None or len(path) <= 0:
            break
        testimg = Image.open(path)
        testvec = get_cacched_histvecctor(path)
        resultdict = {}
        for imagepath in tqdm(imagepaths):
            vec = get_cacched_histvecctor(imagepath)
            similarity = 1 - cosine(vec, testvec)
            resultdict[imagepath] = similarity

        for imgpath, value in sorted(resultdict.items(), key=itemgetter(1))[-n:]:
            basefilename = os.path.basename(imgpath)
            print('{}: {}'.format(basefilename, value))