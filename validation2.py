import argparse
import os
import torch
import numpy as np
import cv2
import hdf5storage as hdf5
import time

import utils
import dataset

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--pre_train', type = bool, default = True, help = 'pre_train or not')
    parser.add_argument('--test_batch_size', type = int, default = 1, help = 'size of the testing batches for single GPU')
    parser.add_argument('--num_workers', type = int, default = 2, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--val_path', type = str, default = './validation', help = 'saving path that is a folder')
    parser.add_argument('--task_name', type = str, default = 'track2', help = 'task name for loading networks, saving, and log')
    # Network initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--activ', type = str, default = 'lrelu', help = 'activation type of networks')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = 'input channels for generator')
    parser.add_argument('--out_channels', type = int, default = 31, help = 'output channels for generator')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for generator')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of generator')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of generator')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = '../NTIRE2020_Validation_RealWorld', help = 'baseroot')
    # NTIRE2020_Validation_Clean    NTIRE2020_Validation_RealWorld
    opt = parser.parse_args()

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    load_net_name_list = []
    load_net_name_list.append('./track2/code1_320_G_epoch7000_bs4.pth')
    load_net_name_list.append('./track2/code1_384_G_epoch6000_bs4.pth')
    load_net_name_list.append('./track2/code1_bs2_G_epoch6000_bs2.pth')
    load_net_name_list.append('./track2/code1_bs4_G_epoch6000_bs4.pth')
    load_net_name_list.append('./track2/code1_first_G_epoch5000_bs8.pth')
    load_net_name_list.append('./track2/code1_G_epoch10000_bs8.pth')
    load_net_name_list.append('./track2/code1_second_G_epoch4000_bs8.pth')
    load_net_name_list.append('./track2/code2_G_epoch6000_bs8.pth')

    for i, name in enumerate(load_net_name_list):
        if i < 7:
            # Initialize
            generator = utils.create_generator_val1(opt, name).cuda()
            test_dataset = dataset.HS_multiscale_ValDSet(opt)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = opt.test_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
            sample_folder = os.path.join(opt.val_path, opt.task_name, str(i))
            utils.check_path(sample_folder)
            print('Network name: %s; The %d-th iteration' % (name, i))

            # forward
            for j, (img1, img2, path) in enumerate(test_loader):
                # To device
                img1 = img1.cuda()
                img2 = img2.cuda()
                path = path[0]
                print(opt.task_name, i, path)

                # Forward propagation
                with torch.no_grad():
                    out1 = generator(img1) # [0:480, 0:512, :], [1, 31, 480, 512]
                    out2 = generator(img2) # [2:482, 0:512, :], [1, 31, 480, 512]
                out = torch.cat((out1, out2[:, :, 478:, :]), 2)

                # Save
                out = out.clone().data.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, :].astype(np.float64)
                save_img_name = path[:12] + '.mat'
                save_img_path = os.path.join(sample_folder, save_img_name)
                hdf5.write(data = out, path = 'cube', filename = save_img_path, matlab_compatible = True)
        
        if i == 7:
            # Initialize
            generator = utils.create_generator_val2(opt, name).cuda()
            test_dataset = dataset.HS_multiscale_ValDSet(opt)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = opt.test_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
            sample_folder = os.path.join(opt.val_path, opt.task_name, str(i))
            utils.check_path(sample_folder)
            print('Network name: %s; The %d-th iteration' % (name, i))

            # forward
            for j, (img1, img2, path) in enumerate(test_loader):
                # To device
                img1 = img1.cuda()
                img2 = img2.cuda()
                path = path[0]
                print(opt.task_name, i, path)

                # Forward propagation
                with torch.no_grad():
                    out1 = generator(img1) # [0:480, 0:512, :], [1, 31, 480, 512]
                    out2 = generator(img2) # [2:482, 0:512, :], [1, 31, 480, 512]
                out = torch.cat((out1, out2[:, :, 478:, :]), 2)

                # Save
                out = out.clone().data.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, :].astype(np.float64)
                save_img_name = path[:12] + '.mat'
                save_img_path = os.path.join(sample_folder, save_img_name)
                hdf5.write(data = out, path = 'cube', filename = save_img_path, matlab_compatible = True)
