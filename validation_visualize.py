import argparse
import os
import torch
import numpy as np
import cv2

import utils
import dataset

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--pre_train', type = bool, default = True, help = 'pre_train or not')
    parser.add_argument('--load_name', type = str, default = './track1/code1_G_epoch9000_bs8.pth', \
            help = 'load the pre-trained model with certain epoch, None for pre-training')
    parser.add_argument('--test_batch_size', type = int, default = 1, help = 'size of the testing batches for single GPU')
    parser.add_argument('--num_workers', type = int, default = 2, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--val_path', type = str, default = './validation_visualization', help = 'saving path that is a folder')
    parser.add_argument('--task_name', type = str, default = 'track1', help = 'task name for loading networks, saving, and log')
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
    parser.add_argument('--baseroot', type = str, default = './NTIRE2020_Test_Clean', help = 'baseroot')
    opt = parser.parse_args()

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    # Initialize
    generator = utils.create_generator(opt).cuda()
    test_dataset = dataset.HS_multiscale_ValDSet(opt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = opt.test_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    sample_folder = os.path.join(opt.val_path, opt.task_name)
    utils.check_path(sample_folder)

    # forward
    for i, (img1, img2, path) in enumerate(test_loader):
        # To device
        img1 = img1.cuda()
        img2 = img2.cuda()
        path = path[0]
        print(opt.task_name, path)

        # Forward propagation
        with torch.no_grad():
            out1 = generator(img1) # [0:480, 0:512, :], [1, 31, 480, 512]
            out2 = generator(img2) # [2:482, 0:512, :], [1, 31, 480, 512]
        out = torch.cat((out1, out2[:, :, 478:, :]), 2)

        # Sample data every iter
        out = out * 255
        # Process img_copy and do not destroy the data of img
        out_copy = out.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        out_copy = np.clip(out_copy, 0, 255).astype(np.uint8)
        
        # Save
        temp = out_copy[0, :, :, 0]
        save_img_name = path[:12] + '_0' + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, temp)
        temp = out_copy[0, :, :, 1]
        save_img_name = path[:12] + '_1' + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, temp)
        temp = out_copy[0, :, :, 2]
        save_img_name = path[:12] + '_2' + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, temp)
        temp = out_copy[0, :, :, 10]
        save_img_name = path[:12] + '_10' + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, temp)
        temp = out_copy[0, :, :, 20]
        save_img_name = path[:12] + '_20' + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, temp)
        temp = out_copy[0, :, :, 30]
        save_img_name = path[:12] + '_30' + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, temp)
