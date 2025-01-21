
import os
from argparse import ArgumentParser
from operator import itemgetter

from PIL import Image
from tqdm import tqdm
from scipy.spatial.distance import cosine
from transformers import AutoImageProcessor, HieraModel, ViTModel


def get_argparser():
    argparser = ArgumentParser(description='Sorting image similarity using ViT.')
    argparser.add_argument('image1path', help='path of the reference image')
    argparser.add_argument('imagepaths', nargs='+', help='paths of images to compare')
    argparser.add_argument('--n', default=10, type=int, help='number of files shown. Default: 10)')
    argparser.add_argument('--vit', default=False, action='store_true', help='use ViT model')
    argparser.add_argument('--hiera', default=False, action='store_true', help='use Hiera model')
    return argparser


def get_hiera_image_model():
    image_processor = AutoImageProcessor.from_pretrained('facebook/hiera-large-224-hf')
    model = HieraModel.from_pretrained('facebook/hiera-large-224-hf')
    return image_processor, model


def get_vit_image_model():
    image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    return image_processor, model


def get_image_embedding(image, model, image_processor):
    inputs = image_processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    # inputs = image_processor(images=image)
    # outputs = model(pixel_values=torch.FloatTensor(inputs['pixel_values']))
    return outputs.pooler_output.detach().numpy()[0]


def go_through_images(image, imagepaths, model, image_processor):
    image_embedding = get_image_embedding(image, model, image_processor)
    result_dict = {}
    for imagepath in tqdm(imagepaths):
        this_image = Image.open(imagepath)
        this_image_embedding = get_image_embedding(this_image, model, image_processor)
        similarity = 1 - cosine(image_embedding, this_image_embedding)
        result_dict[imagepath] = similarity

    return result_dict

if __name__ == '__main__':
    args = get_argparser().parse_args()
    assert (args.vit) ^ (args.hiera)

    image = Image.open(args.image1path)
    if args.vit:
        image_processor, model = get_vit_image_model()
    else:
        image_processor, model = get_hiera_image_model()
    result_dict = go_through_images(image, args.imagepaths, model, image_processor)

    for imgpath, value in sorted(result_dict.items(), key=itemgetter(1), reverse=True)[-args.n:]:
        basefilename = os.path.basename(imgpath)
        print('{}: {}'.format(basefilename, value))
