import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
import re
from packaging import version

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from dataset.dark_zurich_dataset import DarkZurichDataSet
import os
from PIL import Image
from utils.tool import fliplr
import matplotlib.pyplot as plt
import torch.nn as nn
import yaml
import imageio as iio

torch.backends.cudnn.benchmark=True

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './data/Cityscapes/data'
DATA_LIST_PATH = './dataset/cityscapes_list/train.txt'
SAVE_PATH = './data/Dark_zurich/data/pseudo_ohl-1/test'

if not os.path.isdir('./data/Dark_zurich/data/pseudo_ohl-1/'):
    os.makedirs('./data/Dark_zurich/data/pseudo_ohl-1/')
    os.makedirs(SAVE_PATH)

IGNORE_LABEL = 255
NUM_CLASSES = 19
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_FROM_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'
SET = 'train' # We generate pseudo label for training set
INPUT_SIZE = '800,512'

MODEL = 'DeeplabMulti'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--batchsize", type=int, default=4,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    return parser.parse_args()

def save_heatmap(output_name):
    output, name = output_name
    fig = plt.figure()
    plt.axis('off')
    heatmap = plt.imshow(output, cmap='viridis')
    fig.colorbar(heatmap)
    fig.savefig('%s_heatmap.png' % (name.split('.jpg')[0]))
    return

def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    w, h = map(int, args.input_size.split(','))

    config_path = os.path.join(os.path.dirname(args.restore_from),'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)

    args.model = config['model']
    print('ModelType:%s'%args.model)
    print('NormType:%s'%config['norm_style'])
    gpu0 = args.gpu
    batchsize = args.batchsize

    model_name = os.path.basename( os.path.dirname(args.restore_from) )
    #args.save += model_name

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    confidence_path = os.path.join(args.save, 'submit/confidence')
    label_path = os.path.join(args.save, 'submit/labelTrainIds')
    label_invalid_path = os.path.join(args.save, 'submit/labelTrainIds_invalid')
    for path in [confidence_path, label_path, label_invalid_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    if args.model == 'DeepLab':
        model = DeeplabMulti(num_classes=args.num_classes, use_se = config['use_se'], train_bn = False, norm_style = config['norm_style'])
    elif args.model == 'Oracle':
        model = Res_Deeplab(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_ORC
    elif args.model == 'DeeplabVGG':
        model = DeeplabVGG(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_VGG

    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)

    try:
        model.load_state_dict(saved_state_dict)
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(gpu0)

    testloader = data.DataLoader(DarkZurichDataSet(args.data_dir, args.data_list, crop_size=(h, w), resize_size=(w, h), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=4)

    scale = 1.25
    testloader2 = data.DataLoader(DarkZurichDataSet(args.data_dir, args.data_list, crop_size=(round(h*scale), round(w*scale) ), resize_size=( round(w*scale), round(h*scale)), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=4)


    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(1080, 1920), mode='bilinear')

    sm = torch.nn.Softmax(dim = 1)
    log_sm = torch.nn.LogSoftmax(dim = 1)
    kl_distance = nn.KLDivLoss( reduction = 'none')
    prior = np.load('./utils/prior_all.npy').transpose((2,0,1))[np.newaxis, :, :, :]
    prior = torch.from_numpy(prior)
    for index, img_data in enumerate(zip(testloader, testloader2) ):
        batch, batch2 = img_data
        image, _, name = batch
        image2, _, name2 = batch2

        inputs = image.cuda()
        inputs2 = image2.cuda()
        print('\r>>>>Extracting feature...%04d/%04d'%(index*batchsize, args.batchsize*len(testloader)), end='')
        if args.model == 'DeepLab':
            with torch.no_grad():
                output1, output2 = model(inputs)
                output_batch = interp(sm(0.5* output1 + output2))

                heatmap_batch = torch.sum(kl_distance(log_sm(output1), sm(output2)), dim=1)

                output1, output2 = model(fliplr(inputs))
                output1, output2 = fliplr(output1), fliplr(output2)
                output_batch += interp(sm(0.5 * output1 + output2))
                del output1, output2, inputs

                output1, output2 = model(inputs2)
                output_batch += interp(sm(0.5* output1 + output2))
                output1, output2 = model(fliplr(inputs2))
                output1, output2 = fliplr(output1), fliplr(output2)
                output_batch += interp(sm(0.5 * output1 + output2))
                del output1, output2, inputs2
                ratio = 0.95
                output_batch = output_batch.cpu() / 4
                # output_batch = output_batch *(ratio + (1 - ratio) * prior)
                output_batch = output_batch.data.numpy()
                heatmap_batch = heatmap_batch.cpu().data.numpy()
        elif args.model == 'DeeplabVGG' or args.model == 'Oracle':
            output_batch = model(Variable(image).cuda())
            output_batch = interp(output_batch).cpu().data.numpy()

        output_batch = output_batch.transpose(0,2,3,1)
        score_batch = np.max(output_batch, axis=3)
        output_batch = np.asarray(np.argmax(output_batch, axis=3), dtype=np.uint8)

        threshold = 0.3274
        for i in range(output_batch.shape[0]):
            output_single = output_batch[i,:,:]
            output_col = colorize_mask(output_single)
            output = Image.fromarray(output_single)

            name_tmp = name[i].split('/')[-1]
            dir_name = name[i].split('/')[-2]
            save_path = args.save + '/' + dir_name
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            output.save('%s/%s' % (save_path, name_tmp))
            print('%s/%s' % (save_path, name_tmp))
            output_col.save('%s/%s_color.png' % (save_path, name_tmp.split('.')[0]))

            # heatmap_tmp = heatmap_batch[i,:,:]/np.max(heatmap_batch[i,:,:])
            # fig = plt.figure()
            # plt.axis('off')
            # heatmap = plt.imshow(heatmap_tmp, cmap='viridis')
            # fig.colorbar(heatmap)
            # fig.savefig('%s/%s_heatmap.png' % (save_path, name_tmp.split('.')[0]))

            if args.set == 'test' or args.set == 'val':
                # label
                output.save('%s/%s' % (label_path, name_tmp))
                # label invalid
                output_single[score_batch[i, :, :] < threshold] = 255
                output = Image.fromarray(output_single)
                output.save('%s/%s' % (label_invalid_path, name_tmp))
                # conficence

                confidence = score_batch[i, :, :] * 65535
                confidence = np.asarray(confidence, dtype=np.uint16)
                print(confidence.min(), confidence.max())
                iio.imwrite('%s/%s' % (confidence_path, name_tmp), confidence)

    return args.save

if __name__ == '__main__':
    with torch.no_grad():
        save_path = main()
    #os.system('python compute_iou.py ./data/Cityscapes/data/gtFine/train %s'%save_path)
