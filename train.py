

import os, argparse, time, sys, stat, shutil
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from util.MF_dataset import MF_dataset
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from util.util import compute_results, visualize
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from model import RTFNet

# config
n_class   = 13
data_dir  = './data/'
weight_dir = './weights/'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0),
]


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def train(epo, model, train_loader, optimizer):
    lr_this_epo = args.lr_start * args.lr_decay ** (epo)
    print('lr this epo %s is %s' % (epo, lr_this_epo))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_epo
    start_t = time.time()
    model.train()

    for it, (images, labels, names) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        if args.gpu >= 0:
            images = images.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        if (it+1) % 10 == 0:
            print('|-epo %s/%s, train iter %s/%s, %.2f img/sec loss: %.4f' \
                % (epo, args.epoch_max, it+1, train_loader.n_iter, (it+1)*args.batch_size/(time.time()-start_t), float(loss)))

        # for tensorboard
        total_it = epo * train_loader.n_iter + it

        if total_it % 1 == 0:
            writer.add_scalar('Train/loss', loss, total_it)

        view_figure = True # note that I have not colorized the GT and predictions here
        if total_it % 500 == 0:
            if view_figure:
                input_rgb_images = vutils.make_grid(images[:,:3], nrow=8, padding=10) # can only display 3-channel images, so images[:,:3]
                writer.add_image('Train/input_rgb_images', input_rgb_images, total_it)

                scale = max(1, 255//n_class) # label (0,1,2..) is invisable, multiply a constant for visualization
                groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)  # change to 3-channel for visualization
                groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                writer.add_image('Train/groudtruth_images', groudtruth_images, total_it)

                predicted_tensor = logits.argmax(1).unsqueeze(1) * scale # mini_batch*n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor),1) # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                writer.add_image('Train/predicted_images', predicted_images, total_it)


def validation(epo, model, val_loader):

    start_t = time.time()
    model.eval()

    with torch.no_grad():
        for it, (images, labels, names) in enumerate(val_loader):
            images = Variable(images)
            labels = Variable(labels)
            if args.gpu >= 0:
                images = images.cuda(args.gpu)
                labels = labels.cuda(args.gpu)

            logits = model(images) 
            loss = F.cross_entropy(logits, labels)
            if (it+1) % 10 == 0:
                # time.time() returns the current time
                print('|-epo %s/%s, val iter %s/%s, %.2f img/sec loss: %.4f' \
                        % (epo, args.epoch_max, it+1, val_loader.n_iter, (it+1)*args.batch_size/(time.time()-start_t), float(loss)))
            if it == 50:
                import cv2
                pred_0 = logits.argmax(1).cpu().numpy()[0]
                cv2.imwrite(names[0]+".png", pred_0)
                # time.time() returns the current time

            # for tensorboard 
            total_it = epo * val_loader.n_iter + it
 
            if total_it % 1 == 0:
                writer.add_scalar('Validation/loss', loss, total_it)

            view_figure = False # note that I have not colorized the GT and predictions here
            if total_it % 100 == 0:
                if view_figure:
                    input_rgb_images = vutils.make_grid(images[:,:3], nrow=8, padding=10) # can only display 3-channel images, so images[:,:3]
                    writer.add_image('Validation/input_rgb_images', input_rgb_images, total_it)

                    scale = max(1, 255//n_class) # label (0,1,2..) is invisable, multiply a constant for visualization
                    groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                    groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)  # change to 3-channel for visualization
                    groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/groudtruth_images', groudtruth_images, total_it)

                    predicted_tensor = logits.argmax(1).unsqueeze(1) * scale # mini_batch*n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                    predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor),1) # change to 3-channel for visualization, mini_batch*1*480*640
                    predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/predicted_images', predicted_images, total_it)


def testing(epo, model, test_loader):

    model.eval()
    conf_total = np.zeros((n_class, n_class))

    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images)
            labels = Variable(labels)
            if args.gpu >= 0:
                images = images.cuda(args.gpu)
                labels = labels.cuda(args.gpu)

            logits = model(images)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(label, prediction, [0,1,2,3,4,5,6,7,8,9,10,11,12]) # conf is n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            if (it+1) % 10 == 0:
                print('|-epo %s/%s. testing iter %s/%s.' % (epo, args.epoch_max, it+1, test_loader.n_iter))

    precision, recall, IoU, Acc= compute_results(conf_total)


    writer.add_scalar('Test/average_precision',precision.mean(), epo)
    writer.add_scalar('Test/average_recall', recall.mean(), epo)
    writer.add_scalar('Test/average_IoU', IoU.mean(), epo)
    writer.add_scalar('Test/average_Acc', Acc.mean(), epo)

    return [np.mean(np.nan_to_num(precision)), np.mean(np.nan_to_num(recall)), np.mean(np.nan_to_num(IoU)), np.mean(np.nan_to_num(Acc))],precision,recall,IoU,Acc

def main():

    train_dataset = MF_dataset(data_dir, 'train', have_label=True, transform=augmentation_methods)
    val_dataset  = MF_dataset(data_dir, 'val', have_label=True) 
    test_dataset = MF_dataset(data_dir, 'test', have_label=True)

    train_loader  = DataLoader(
        dataset     = train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    val_loader  = DataLoader(
        dataset     = val_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    test_loader = DataLoader(
        dataset      = test_dataset,
        batch_size   = args.batch_size,
        shuffle      = False,
        num_workers  = args.num_workers,
        pin_memory   = True,
        drop_last    = False
    )
    train_loader.n_iter = len(train_loader)
    val_loader.n_iter = len(val_loader)
    test_loader.n_iter = len(test_loader)
    #model.load_state_dict(torch.load('E:/PyCharm_Proj/RTF-net/weights/RTFNet/45.pth'))  #重加载模型
    testing_results_file = os.path.join(weight_dir, 'testing_results_file.txt')
    with open(testing_results_file, 'w') as testing_result_appender:
        testing_result_appender.write("# epoch,    ave_precision  ,          ave_IOU   ,          ave_Acc   .\n")

    for epo in range(args.epoch_from, args.epoch_max):
        print('\ntrain %s, epo #%s begin...' %(args.model_name, epo))
        train(epo, model, train_loader, optimizer_SGD)
        validation(epo, model, val_loader)
 
        checkpoint_model_file = os.path.join(weight_dir, str(epo)+'.pth')
        print('|- saving check point %s: ' %checkpoint_model_file)
        if epo % 5 == 0:
            torch.save(model.state_dict(), checkpoint_model_file)

        np.set_printoptions(precision=4, threshold=np.inf, linewidth=np.inf, suppress=True, floatmode='fixed')
        testing_results = testing(epo, model, test_loader)
        precision, recall, IoU, Acc = testing_results[0],testing_results[1],testing_results[2],testing_results[3]
        with open(testing_results_file, 'a') as testing_result_appender:
            testing_result_appender.write(str(epo) + ', ' + str(testing_results[0]) + ',' + str(testing_results[1]) + ',' + str(testing_results[2]) +',' + str(testing_results[3]))
            testing_result_appender.write('\n')
            for i in range(len(precision)):
                testing_result_appender.write(precision[i] + "  ")
            testing_result_appender.write('\n')
            for i in range(len(recall)):
                testing_result_appender.write(recall[i] + "  ")
            testing_result_appender.write('\n')
            for i in range(len(IoU)):
                testing_result_appender.write(IoU[i] + "  ")
            testing_result_appender.write('\n')

        # print('|- saving testing results.\n')
 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train with pytorch')
    ############################################################################################# 
    parser.add_argument('--model_name',  '-M',  type=str, default='RTFNet')
    #batch_size: RTFNet-152: 2; RTFNet-101: 2; RTFNet-50: 3; RTFNet-34: 10; RTFNet-18: 15;
    parser.add_argument('--batch_size',  '-B',  type=int, default=1)
    parser.add_argument('--lr_start',  '-LS',  type=float, default=0.05)  
    parser.add_argument('--gpu',        '-G',  type=int, default=0)
    ############################################################################################
    parser.add_argument('--lr_decay', '-LD', type=float, default=0.98)
    parser.add_argument('--epoch_max' ,  '-E',  type=int, default=301)  # please stop training mannully
    parser.add_argument('--epoch_from',  '-EF', type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=0)
    args = parser.parse_args()
 
    #torch.cuda.set_device(args.gpu)
    print("\nthe gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    model = eval(args.model_name)(n_class=n_class)
    model = torch.nn.DataParallel(model).cuda()
    print(get_parameter_number(model))
    #if args.gpu >= 0: model.cuda(args.gpu)
    optimizer_SGD = torch.optim.SGD(model.parameters(), lr=args.lr_start, momentum=0.9)
    optimizer_ADAM = torch.optim.Adam(model.parameters(), lr=args.lr_start, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # optimizer = torch.optim.adagrad(model.parameters(), lr=args.lr_start, weight_decay=0)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr_start, rho=0.9, eps=1e-06, weight_decay=0)
    weight_dir = os.path.join(weight_dir, args.model_name)
    if os.path.exists(weight_dir) is True:
        pass
        # print('previous weights folder exist, will delete the weights folder')
        # shutil.rmtree(weight_dir)
        # os.makedirs(weight_dir)
        # os.chmod(weight_dir, stat.S_IRWXO)  # allow the folder created by docker read and written by local machine
    else:
        os.makedirs(weight_dir)
        os.chmod(weight_dir, stat.S_IRWXO)

    if os.path.exists("runs") is True:
        pass
        print('previous runs folder exist, will delete the runs folder')
        # shutil.rmtree("runs")
        # os.makedirs("runs")
        # os.chmod("runs", stat.S_IRWXO)
    else:
        os.makedirs('runs')
        os.chmod('runs', stat.S_IRWXO)

    # tensorboardX setup
    writer = SummaryWriter('runs')  # default log directory is 'runs'

    print('training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('weight will be saved in: %s' % weight_dir)

    main()
