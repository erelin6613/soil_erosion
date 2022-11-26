# data loader
from __future__ import print_function, division
import os
import glob
import random
import math

import torch
from skimage import io, transform, color
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

import sys

sys.path.append(os.path.abspath(os.getcwd()))
random.seed(999)
default_size = 320

#from midas_model.midas.transforms import Resize, NormalizeImage, PrepareForNet
#from midas_model.midas.midas_net_custom import MidasNet_small
#from midas_model.utils import read_image


depth_model_type = "midas_v21_small"
depth_model_path = "midas_v21_small-70d6b9c8.pt"

# if depth_model_type == "midas_v21_small":
#     depth_model = MidasNet_small(
#         depth_model_path,
#         features=64,
#         backbone="efficientnet_lite3",
#         exportable=True,
#         non_negative=True,
#         blocks={'expand': True})
#     net_w, net_h = 320, 320
#     resize_mode="upper_bound"
#     normalization = NormalizeImage(
#         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#     )

#==========================dataset load==========================


def find_q1_q3(array):
    q1, q3 = np.quantile(array, 0.25), np.quantile(array, 0.75)
    return q1, q3

def make_edges(image):
    th1, th2 = find_q1_q3(image)
    kernel = np.ones((5, 5), dtype=np.uint8)
    edges = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return edges

def prepare_image(image):
    image = cv2.resize(image, (default_size, default_size), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)


class RescaleT(object):

    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size

    def __call__(self,sample, mask=None):
        imidx, image, edges, depth, label = sample['imidx'], sample['image'], sample['edges'], sample['depth'], sample['label']
        h, w = np.array(image).shape[:2]

        if isinstance(self.output_size,int):
            if h > w:
                new_h, new_w = self.output_size*h/w,self.output_size
            else:
                new_h, new_w = self.output_size,self.output_size*w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        resize = transforms.Resize(size = (self.output_size,self.output_size), interpolation=Image.NEAREST)
        lbl = resize(label)
        img = resize(image)
        edges = resize(edges)
        depth = resize(depth)

        return {'imidx':imidx, 'image':img, 'edges': edges, 'depth': depth, 'label':lbl}


class Rescale(object):

    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size

    def __call__(self,sample):
        imidx, image, edges, depth, label = sample['imidx'], sample['image'], sample['edges'], sample['depth'], sample['label']

        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]
            edges = edges[::-1]
            depth = depth[::-1]

        h, w = image.shape[:2]

        if isinstance(self.output_size,int):
            if h > w:
                new_h, new_w = self.output_size*h/w,self.output_size
            else:
                new_h, new_w = self.output_size,self.output_size*w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h,new_w), mode='constant')
        lbl = transform.resize(label, (new_h,new_w), mode='constant', order=0, preserve_range=True)
        edges = transform.resize(edges, (new_h,new_w), mode='constant', order=0, preserve_range=True)
        depth = transform.resize(depth, (new_h, new_w), mode='constant', order=0, preserve_range=True)

        return {
            'imidx':imidx,
            'image':img,
            'edges': edges,
            'depth': depth,
            'label':lbl}


class RandomCrop(object):

    def __init__(self,output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self,sample):
        imidx, image, edges, depth, label = sample['imidx'], np.array(
            sample['image']), np.array(sample['edges']), np.array(
            sample['depth']), np.array(sample['label'])

        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]
            edges = edges[::-1]
            depth = depth[::-1]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]
        edges = edges[top: top + new_h, left: left + new_w]
        depth = depth[top: top + new_h, left: left + new_w]

        return {
            'imidx':imidx,
            'image':image,
            'label':label,
            'depth': depth,
            'edges': edges}


class HorizontalFlip(object):

    def __init__(self, p=0.5):
        self.prob = p

    def __call__(self,sample):
        imidx, image, edges, depth, label = sample['imidx'], np.array(
            sample['image']), np.array(sample['edges']), np.array(
            sample['depth']), np.array(sample['label'])
        rand = random.random()

        if rand >= self.prob:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
            edges = cv2.flip(edges, 1)
            depth = cv2.flip(depth, 1)

        return {
            'imidx':imidx,
            'image':image,
            'label':label,
            'depth': depth,
            'edges': edges}


class RandomRotate(object):

    def __init__(self, p=0.5):
        self.prob = p

    def __call__(self,sample):
        imidx, image, edges, depth, label = sample['imidx'], np.array(
            sample['image']), np.array(sample['edges']), np.array(
            sample['depth']), np.array(sample['label'])
        rot_times = [1, 2, 3]
        rand = random.random()

        if self.prob >= rand:
            rots = np.random.choice(rot_times, 1)
            image = np.rot90(image, rots)
            label = np.rot90(image, rots)
            edges = np.rot90(edges, rots)
            depth = np.rot90(depth, rots)

        return {
            'imidx': imidx,
            'image': image,
            'label': label,
            'depth': depth,
            'edges': edges}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        imidx, image, edges, depth, label = np.array(sample['imidx']), np.array(
            sample['image']), np.array(sample['edges']), np.array(
            sample['depth']), np.array(sample['label'])
        tmpImg = np.zeros((image.shape[0],image.shape[1],3))
        tmpLbl = np.zeros((label.shape[0], label.shape[1], 1))

        tmpEd = np.zeros((edges.shape[0], edges.shape[1], 1))
        tmpDep = np.zeros((depth.shape[0], depth.shape[1], 1))
        image = image/np.max(image)

        if(np.max(label)<1e-6):
            label = label
            edges = edges
            depth = depth
        else:
            label = label/np.max(label)
            edges = edges/np.max(edges)
            depth = depth/np.max(depth)

        if image.shape[2]==1:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
        else:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
            tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

        # print(label.shape)
        tmpLbl[:,:,0] = label #[:,:,0]

        tmpEd[:,:,0] = edges #[:,:,0]
        tmpDep[:,:,0] = depth #[:,:,0]

        try:
            # print(tmpImg.shape, tmpLbl.shape, tmpEd.shape, tmpDep.shape)
            tmpEd = torch.from_numpy(tmpEd)
            tmpDep = torch.from_numpy(tmpDep)
            # tmpImg = tmpImg.transpose((2, 0, 1))
            # tmpLbl = tmpLbl.transpose((2, 0, 1))
            # tmpEd = tmpEd.transpose((2, 0, 1))
            # tmpDep = tmpDep.transpose((2,0,1))
        except:
            pass

        # tmpImg = tmpImg.transpose((2, 0, 1))
        # print(cv2.imwrite("test.png", (255*tmpDep).astype(np.uint8)))
        # print(cv2.imwrite("test_im.png", (255*image).astype(np.uint8)))
        # tmpLbl = label.transpose((2, 0, 1))
        # tmpEd = edges.transpose((2, 0, 1))
        # tmpDep = depth.transpose((2,0,1))

        # print(tmpImg.shape, tmpLbl.shape, tmpEd.shape, tmpDep.shape)

        return {
            'imidx':torch.from_numpy(imidx).float(),
            'image': torch.from_numpy(tmpImg).float(),
            'edges': tmpEd.float(),
            'depth': tmpDep.float(),
            'label': torch.from_numpy(tmpLbl).float()}


class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self,flag=0):
        self.flag = flag

    def __call__(self, sample):

        imidx, image, edges, depth, label = sample['imidx'], np.array(
            sample['image']), np.array(sample['edges'])[:,:,np.newaxis], np.array(
            sample['depth'])[:,:,np.newaxis],  np.array(
            sample['label'])[:,:,np.newaxis]
        tmpLbl = np.zeros(label.shape)
        tmpEd = np.zeros(edges.shape)
        tmpDep = np.zeros(depth.shape)

        if(np.max(label)<1e-6):
            label = label
            edges = edges
            depth = depth
        else:
            label = label/np.max(label)
            edges = edges/np.max(edges)
            depth = depth/np.max(depth)

        if self.flag == 2: # with rgb and Lab colors
            tmpImg = np.zeros((image.shape[0],image.shape[1],6))
            tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
            if image.shape[2]==1:
                tmpImgt[:,:,0] = image[:,:,0]
                tmpImgt[:,:,1] = image[:,:,0]
                tmpImgt[:,:,2] = image[:,:,0]
            else:
                tmpImgt = image
            tmpImgtl = color.rgb2lab(tmpImgt)

            tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
            tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
            tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
            tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
            tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
            tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

            tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
            tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
            tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
            tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
            tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
            tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

        elif self.flag == 1: #with Lab color
            tmpImg = np.zeros((image.shape[0],image.shape[1],3))

            if image.shape[2]==1:
                tmpImg[:,:,0] = image[:,:,0]
                tmpImg[:,:,1] = image[:,:,0]
                tmpImg[:,:,2] = image[:,:,0]
            else:
                tmpImg = image

            tmpImg = color.rgb2lab(tmpImg)

            tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
            tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
            tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

            tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
            tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
            tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
        else: # with rgb color
            tmpImg = np.zeros((image.shape[0],image.shape[1],3))
            image = image/np.max(image)
            if image.shape[2]==1:
                tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
                tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
            else:
                tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
                tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

        # print(label.shape, edges.shape, depth.shape)
        tmpLbl[:,:,0] = label[:,:,0]
        tmpEd[:,:,0] = edges[:,:,0]
        tmpDep[:,:,0] = depth[:,:,0]
        tmpImg = tmpImg.transpose((2, 0, 1))

        try:
            tmpLbl = tmpLbl.transpose((2, 0, 1))
        except:
            print(edges.shape, label.shape, tmpLbl.shape, tmpEd.shape)
        tmpEd = tmpEd.transpose((2, 0, 1))
        tmpDep = tmpDep.transpose((2, 0, 1))

        # print(np.array(tmpImg).shape, np.array(tmpEd).shape, np.array(tmpDep).shape, np.array(tmpLbl).shape)

        return {
            'imidx': torch.from_numpy(imidx).float(),
            'image': torch.from_numpy(tmpImg).float(),
            'edges': torch.from_numpy(tmpEd).float(),
            'depth': torch.from_numpy(tmpDep).float(),
            'label': torch.from_numpy(tmpLbl).float()}


def make_depth_image(img_path):
    image = read_image(img_path)

    transform = transforms.Compose(
        [
            Resize(net_w, net_h, resize_target=None,
                keep_aspect_ratio=True, ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC),
            normalization,
            PrepareForNet(),
        ]
    )

    img_input = transform({"image": image})["image"]
    # print(img_input.shape)
    # pred = depth_model(torch.tensor(img_input).unsqueeze(0))
    pred = pred / pred.max()
    pred = (255 * pred[0].detach().numpy()).astype(np.uint8)
    return pred

def get_sobel(img_path):
    image = cv2.imread(img_path, 0)
    image = cv2.resize(image, (320, 320), interpolation = cv2.INTER_AREA)
    # print(image.ddepth)
    try:
        grad_x = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    except:
        grad_x = cv2.Sobel(image, cv2.CV_16U, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(image, cv2.CV_16U, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    # grad_x = cv2.cvtColor(grad_x, cv2.COLOR_BGR2GRAY)
    # grad_y = cv2.cvtColor(grad_y, cv2.COLOR_BGR2GRAY)
    return cv2.bitwise_or(grad_x, grad_y)

class SalObjDataset(Dataset):
    def __init__(self,img_name_list,lbl_name_list,transform=None):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,idx):
        image = Image.open(self.image_name_list[idx]).convert("RGB")
        image = np.array(image)
        imname = self.image_name_list[idx]
        imidx = np.array([idx])

        if(0==len(self.label_name_list)):
            label = np.zeros(np.array(image).shape)
            label = label.astype(np.uint8)[0]
            label = Image.fromarray(label).convert("L")
        else:
            label = Image.open(self.label_name_list[idx]).convert("L")

        edges = prepare_image(image)
        edges = make_edges(edges)
        depth_path = self.image_name_list[idx].replace(
            "resized_images", "depths_images")
        # if not os.path.exists(depth_path):
            # depth = make_depth_image(imname)
        depth = get_sobel(imname)
        # print(depth.shape, edges.shape)
        #     try:
        #         cv2.imwrite(depth_path, depth)
        #     except Exception as e:
        #         print(e)
        # else:
        #     depth = cv2.imread(depth_path, 0)

        image = Image.fromarray(image)
        edges = Image.fromarray(edges)
        depth = Image.fromarray(depth)

        sample = {
            'imidx': imidx,
            'image': image,
            'edges': edges,
            'depth': depth,
            'label': label
            }

        if self.transform:
            sample = self.transform(sample)
            if sample == 1:
                print(self.image_name_list[idx])

        # image = np.stack([
        #     sample['image'][0],
        #     sample['image'][1],
        #     sample['image'][2],
        #     sample['depth'][0],
        #     sample['edges'][0]], 0)

        if len(sample['depth'].shape) == 2:
            sample['depth'] = sample['depth'].unsqueeze(0)

        image = np.stack([
            sample['image'][0],
            sample['image'][1],
            sample['image'][2],
            sample['depth'][0],
            sample['edges'][0]], 0)

        sample = {
            'imidx': imidx,
            'image': image,
            'label': sample['label']
            }

        sample["image"] = image

        return sample

# if __name__ == '__main__':
#     # image = cv2.imread()
#     pred = make_depth_image("datasets/resized_images/0dba989b67a3ec3b6261220db0173472.jpg")
#     # print(pred.shape)
#     pred = pred / pred.max()
#     pred = (255 * pred[0].detach().numpy()).astype(np.uint8)
#     cv2.imwrite("test.png", pred)
