# coding:utf-8
# By Yuxiang Sun, Aug. 2, 2019
# Email: sun.yuxiang@outlook.com

import os, shutil, stat
import argparse 
import numpy as np
import sys
from PIL import Image
import torch 
from torch.autograd import Variable
from util.util import visualize
from model import RTFNet  
import cv2

n_class = 13
image_dir = './data/'
model_dir = './weights/'

def main():

    model = eval(args.model_name)(n_class=n_class)
    if args.gpu >= 0: model.cuda(args.gpu)
    print('| loading model file %s' % model_file)

    pretrained_weight = torch.load(model_file, map_location = lambda storage, loc: storage.cuda(args.gpu))
    own_state = model.state_dict()
    # model.load_state_dict(torch.load('./weights/RTFNet/100.pth'))  # 重加载模型
    for name, param in pretrained_weight.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)
    print('done!')

    file = open('./data/test.txt')
    for line in file.readlines():
        line = str(line).split('\n')[0]
        imgname = line + '.png'
        # thermal = np.asarray(Image.open(image_dir + 'fl_ir_aligned/fl_ir_aligned_' + imgname)).reshape(320,1920)


        #
        # RGB = np.asarray(Image.open(image_dir + 'fl_rgb/fl_rgb_' + imgname)).reshape(320,1920)
        # fuse = np.asarray(Image.open(image_dir + 'fl_rgb/fl_rgb_' + imgname)).reshape(320,1920)


        # thermal = np.asarray(Image.open(image_dir + 'fl_ir_aligned/fl_ir_aligned_' + imgname))
        # print(thermal.size)
        # thermal = np.asarray(Image.fromarray(thermal).resize((1920, 320)), dtype=np.float32)
        #
        # RGB = np.asarray(Image.open(image_dir + 'fl_rgb/fl_rgb_' + imgname))
        # RGB = np.asarray(Image.fromarray(thermal).resize((1920, 320)), dtype=np.float32)


        # RGB = cv2.resize(cv2.imread(image_dir + 'fl_rgb/fl_rgb_' + imgname),(960, 320))
        # thermal = cv2.resize(cv2.imread(image_dir + 'fl_ir_aligned/fl_ir_aligned_' + imgname), (960, 320))

        thermal = np.asarray(Image.open(image_dir + 'fl_ir_aligned/fl_ir_aligned_' + imgname))
        thermal = np.asarray(Image.fromarray(thermal).resize((960, 320)), dtype=np.float32)

        RGB = np.asarray(Image.open(image_dir + 'fl_rgb/fl_rgb_' + imgname))
        RGB = np.asarray(Image.fromarray(RGB).resize((960, 320)), dtype=np.float32)

        image = np.concatenate((RGB, thermal[:, :, 0:1], thermal[:, :, 0:1]), axis=2)
        # image = np.concatenate((image, fuse[:, :, 0:1]), axis=2)

        image = torch.from_numpy(image).float()
        image.unsqueeze_(0)
        # print(image.shape)  # (1, 480, 640, 5)
        image = np.asarray(image, dtype=np.float32).transpose((0, 3, 1, 2)) / 255.0
        print(image.shape)  # (1, 480, 640, 5)
        image = Variable(torch.tensor(image))
        if args.gpu >= 0: image = image.cuda(args.gpu)

        model.eval()
        with torch.no_grad():
            logits = model(image)
            predictions = logits.argmax(1)
            print(predictions.size())
            visualize(imgname, predictions, args.weight_name)

        print('| %s:%s, prediction of %s has been saved in ./demo_results' % (args.model_name, args.weight_name, imgname))
    file.close()



if __name__ == "__main__": 

    if os.path.exists('./demo_results') is True:
        # print('| previous \'./demo_results\' exist, will delete the folder')
        # shutil.rmtree('./demo_results')
        # os.makedirs('./demo_results')
        os.chmod('./demo_results', stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine
    else:
        os.makedirs('./demo_results')
        os.chmod('./demo_results', stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine

    parser = argparse.ArgumentParser(description='Run demo with pytorch')
    parser.add_argument('--model_name', '-M',  type=str, default='RTFNet')
    parser.add_argument('--weight_name', '-W', type=str, default='RTFNet')  # RTFNet_152, RTFNet_50, please change the number of layers in the network file
    parser.add_argument('--gpu',        '-G',  type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    print("\n| the gpu count:", torch.cuda.device_count())
    print("| the current used gpu:", torch.cuda.current_device(), '\n')

    model_dir = os.path.join(model_dir, args.weight_name)  # model_dir = './weights_backup/'

    if os.path.exists(model_dir) is False:
        print("| the %s does not exit." %(model_dir))
        sys.exit()
    model_file = os.path.join(model_dir, '95.pth')  #

    if os.path.exists(model_file) is True:
        print('| use the final model file.')
    else:
        print('| no model file found.')
        sys.exit()
    print('| running %s:%s demo on GPU #%d with pytorch' % (args.model_name, args.weight_name, args.gpu))
    main()
