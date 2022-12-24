from __future__ import print_function
import argparse
import os
import random
import torch

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import math

from models.P3SNet import P3SNet,P3SNET_plus


parser = argparse.ArgumentParser(description='P3SNET')
parser.add_argument('--model', default='P3SNET', help='select model : P3SNET, P3SNET_plus')
parser.add_argument('--dataset', default='SceneFlow',help='Dataset: SceneFlow, KITTI2012, KITTI2015')
parser.add_argument('--datapath', default='/media/alper/DOPOSSD1/dataset/', help='datapath') #./dataset/SceneFlowData/
parser.add_argument('--maxdisp', type=int ,default=192, help='maxium disparity')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train')
parser.add_argument('--train_batch_size', type=int ,default=2, help='train_batch_size')
parser.add_argument('--test_batch_size', type=int ,default=2, help='test_batch_size')
parser.add_argument('--loadmodel', default= None, help='load model')
parser.add_argument('--save_dir', default='./saved_models',help='save model')
parser.add_argument('--mode', type=str, default='train', help='mode: train, eval')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')


args = parser.parse_args()
args.cuda =  torch.cuda.is_available()
mode = args.mode
train_batch_size = args.train_batch_size
test_batch_size = args.test_batch_size

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.dataset == 'KITTI2015':
    from dataloader import KITTIloader2015 as lt
    from dataloader import KITTILoader as DA
elif args.dataset == 'KITTI2012':
    from dataloader import KITTIloader2012 as lt
    from dataloader import KITTILoader as DA
else:
    from dataloader import listflowfile as lt
    from dataloader import SecenFlowLoader as DA

if args.model == "P3SNET":
    weights_outputs = [4.0, 3.0, 2.0, 1.0]
    number_scale = 3
else: # P3SNET+
    weights_outputs = [5.0, 4.0, 3.0, 2.0, 1.0]
    number_scale = 4

weights_outputs = [i/np.sum(weights_outputs) for i in weights_outputs]
print("[***] weights_outputs : ", weights_outputs)

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp  = lt.dataloader(args.datapath)
TrainImgLoader = torch.utils.data.DataLoader(DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True), batch_size= train_batch_size, shuffle= True, num_workers=8, drop_last=False)
TestImgLoader = torch.utils.data.DataLoader(DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False), batch_size= test_batch_size, shuffle= False, num_workers=8, drop_last=False)

model = P3SNET_plus(args.maxdisp,number_scale)


if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

def train(imgL,imgR, disp_L):
        model.train()

        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

        mask = disp_true < args.maxdisp
        mask.detach_()
        optimizer.zero_grad()

        disp_results = model(imgL,imgR)

        disp0 =  disp_results[0]
        loss = weights_outputs[0]*F.smooth_l1_loss(disp0[mask], disp_true[mask], reduction='mean')
        for i in range(1, 4):
            loss += weights_outputs[i]*F.smooth_l1_loss(disp_results[i][mask], disp_true[mask], reduction='mean')

        loss.backward()
        optimizer.step()

        return loss.data

def test(imgL,imgR,disp_true):

        model.eval()

        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

        mask = disp_true < 192
        immood = int(np.power(2, number_scale+1))
        if imgL.shape[2] % immood != 0:
            times = imgL.shape[2] // immood
            top_pad = (times+1)*immood -imgL.shape[2]
        else:
            top_pad = 0

        if imgL.shape[3] % immood != 0:
            times = imgL.shape[3]//immood
            right_pad = (times+1)*immood-imgL.shape[3]
        else:
            right_pad = 0

        imgL = F.pad(imgL,(0,right_pad, top_pad,0))
        imgR = F.pad(imgR,(0,right_pad, top_pad,0))

        with torch.no_grad():
            output3 = model(imgL,imgR)
            output3 = torch.squeeze(output3)

        if top_pad !=0:
            img = output3[:,top_pad:,:]
        else:
            img = output3

        if len(disp_true[mask])==0:
           loss = 0
        else:
           loss = F.l1_loss(img[mask],disp_true[mask]) #torch.mean(torch.abs(img[mask]-disp_true[mask]))  # end-point-error

        return loss.data.cpu()

old_lr = 0.01

def adjust_learning_rate(optimizer, epoch):
    global old_lr
    lr = old_lr
    if epoch < 100 :
        lr = 0.01
    else:
        if epoch % 50 == 0:
           lr = old_lr/2
           old_lr = lr
           print("lr: ",lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    for epoch in range(1, args.epochs+1):
        print('This is %d-th epoch' %(epoch))
        total_train_loss = 0
        print_loss = 0
        adjust_learning_rate(optimizer,epoch)

        #------------- TRAIN ------------------------------------------------------------
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):

            loss = train(imgL_crop,imgR_crop, disp_crop_L)
            total_train_loss += loss
            print_loss += loss
            if batch_idx %100 ==0:
                print('Iter %d training loss = %.3f' %(batch_idx, print_loss/100))
                print_loss = 0

        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))
        #----------------------------------------------------------------------------------

        #------------- TEST ------------------------------------------------------------
        total_test_loss = 0
        for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
               test_loss = test(imgL,imgR, disp_L)
               #print('Iter %d test loss = %.3f' %(batch_idx, test_loss))
               total_test_loss += test_loss

        print('total test loss = %.3f' %(total_test_loss/len(TestImgLoader)))
        #----------------------------------------------------------------------------------

        #SAVE
        savefilename = args.save_dir+'/checkpoint_'+str(epoch)+'.tar'
        torch.save({'epoch': epoch,'state_dict': model.state_dict(),'train_loss': total_train_loss/len(TrainImgLoader),}, savefilename)

def eval():

    #------------- TEST ------------------------------------------------------------
    total_test_loss = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
           test_loss = test(imgL,imgR, disp_L)
           total_test_loss += test_loss
    print('total test loss = %.3f' %(total_test_loss/len(TestImgLoader)))
    #----------------------------------------------------------------------------------


if __name__ == '__main__':

    if mode == "train":
        main()
    elif mode == "eval":
        eval();
