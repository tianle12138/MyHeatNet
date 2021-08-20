import cv2
import os
image_dir = './data/'
file = open('./data/val.txt')

for line in file.readlines():
    line = str(line).split('\n')[0]
    imgname = line + '.png'
    print(imgname)
#     org = cv2.imread(image_dir + 'RGB/' + imgname)
#     pred = cv2.imread('demo_results/pred_RTFNet_' + imgname)
#     result = cv2.addWeighted(org, 0.5, pred, 0.5, 0)
#
#     if os.path.exists('./fuse_pred') is False:
#         os.makedirs('./fuse_pred')
#     cv2.imwrite('./fuse_pred/' + imgname, result)
# file.close()


#
# for line in file.readlines():
#     line = str(line).split('\n')[0]
#     imgname = line + '.png'
#     org = cv2.imread(image_dir + 'RGB/' + imgname)
#     pred = cv2.imread('demo_results/pred_RTFNet_' + imgname)
#     result = cv2.addWeighted(org, 0.5, pred, 0.5, 0)
#
#     if os.path.exists('./fuse_pred') is False:
#         os.makedirs('./fuse_pred')
#     cv2.imwrite('./fuse_pred/' + imgname, result)
# file.close()
