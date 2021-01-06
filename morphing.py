import sys

import argparse
from argparse import Namespace
import os
from os import listdir
from os.path import basename, isfile, join
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from numba import jit
# from PIL import Image
from tqdm import tqdm
# from torchvision.utils import save_image
from torchvision.utils import make_grid

from train import calc_loss


# Compute the L2 metric for the transportation cost. Can probably be vectorized to run faster.
@jit("float64[:,:](int64,int64,int64[:,:,:])",nopython=True)
def _generate_metric(height, width, grid):
    # Could probably inpmprove runtime using vectorized code
    C = np.zeros((height*width, height*width))
    i = 0
    j = 0
    for y1 in range(width):
        for x1 in range(height):
            for y2 in range(width):
                for x2 in range(height):
                    C[i,j] = np.square(grid[x1,y1,:] - grid[x2,y2,:]).sum()
                    j += 1
            j = 0
            i += 1
    return C

def generate_metric(im_size: Tuple[int]) -> np.ndarray:
    """
    Computes the Euclidean distances matrix
    
    Arguments:
        im_size {Tuple[int]} -- Size of the input image (height, width)
    
    Returns:
        np.ndarray -- distances matrix
    """
    grid = np.meshgrid(*[range(x) for x in im_size])
    grid = np.stack(grid,-1)
    return _generate_metric(im_size[0], im_size[1], grid)

# Find interpolation given the transportation plan. Can probably be vectorized to run faster.
@jit("float64[:,:](int64,int64,float64[:,:,:,:],float32)",nopython=True)
def generate_interpolation(height, width, plan, t):
    c = np.zeros((height+1, width+1))
    for y1 in range(width):
        for x1 in range(height):
            for y2 in range(width):
                for x2 in range(height):
                    new_loc_x = (1-t)*x1 + t*x2
                    new_loc_y = (1-t)*y1 + t*y2
                    p = new_loc_x - int(new_loc_x)
                    q = new_loc_y - int(new_loc_y)
                    c[int(new_loc_x),int(new_loc_y)] += (1-p)*(1-q)*plan[x1,y1,x2,y2]
                    c[int(new_loc_x)+1,int(new_loc_y)] += p*(1-q)*plan[x1,y1,x2,y2]
                    c[int(new_loc_x),int(new_loc_y)+1] += (1-p)*q*plan[x1,y1,x2,y2]
                    c[int(new_loc_x)+1,int(new_loc_y)+1] += p*q*plan[x1,y1,x2,y2]
    c = c[:height,:width] #* (I1_count*(1-t) + I2_count*t)
    return c

def sinkhorn(a: np.ndarray, b: np.ndarray, C: np.ndarray, height: int, width: int, 
             epsilon: float, threshold: float=1e-7) -> np.ndarray:
    """Computes the sinkhorn algorithm naively, using the CPU.
    
    Arguments:
        a {np.ndarray} -- the first distribution (image), normalized, and shaped to a vector of size height*width.
        b {np.ndarray} -- the second distribution (image), normalized, and shaped to a vector of size height*width.
        C {np.ndarray} -- the distances matrix
        height {int} -- image height
        width {int} -- image width
        epsilon {float} -- entropic regularization parameter
    
    Keyword Arguments:
        threshold {float} -- convergence threshold  (default: {1e-7})
    
    Returns:
        np.ndarray -- the entropic regularized transportation plan, pushing distribution a to b.
    """
    K = np.exp(-C/epsilon)
    v = np.random.randn(*a.shape)
    i = 0
    while True:
        u = a/(K.dot(v))
        v = b/(K.T.dot(u))
        i += 1
        if i % 50 == 0:
            convergence = np.square(np.sum(u.reshape(-1, 1) * K * v.reshape(1,-1), axis=1) - a).sum()
            if convergence < threshold:
                print(f"Iteration {i}. Sinkhorn convergence: {convergence:.2E} (Converged!)")
                break
            else:
                print(f"Iteration {i}. Sinkhorn convergence: {convergence:.2E} ( > {threshold})")

    P = u.reshape(-1, 1) * K * v.reshape(1,-1)
    P = P.reshape(height, width, height, width)
    return P

def preprocess_Q(Q: np.ndarray, max_val: float=None, Q_counts: np.ndarray=None) -> Tuple[np.ndarray, float, np.ndarray]:
    """ Preprocess (normalize) input images before computing their barycenters
    
    Arguments:
        Q {np.ndarray} -- Input images. Every image should reshaped to a column in Q.
    
    Keyword Arguments:
        max_val {float} -- The maximum value. Should be changed from None when using the iterative algorithm (more than 1 iteration in the Algorithm) (default: {None})
        Q_counts {np.ndarray} -- The sum of all the pixel values in each image. Should be changed from None when using the iterative algorithm (more than 1 iteration in the Algorithm) (default: {None})
    
    Returns:
        Tuple[np.ndarray, float, np.ndarray] -- The normalized images the total maximum value and sum of pixels in each image
    """
    if max_val is None:
        max_val = Q.max()
    Q = max_val - Q
    if Q_counts is None:
        Q_counts = np.sum(Q, axis=1, keepdims=True)
    Q = Q / Q_counts
    return Q, max_val, Q_counts

############################################################
#  Modify starting below #
############################################################

def project_on_generator(G: nn.Module, args: Namespace,
                         target_image: np.ndarray, dcgan_img_size: int=64, 
                         ) -> Tuple[np.ndarray, torch.Tensor]:
    """Projects the input image onto the manifold span by the GAN. It operates as follows:
    1. reshape and normalize the image
    2. run the encoder to obtain a latent vector
    3. run the DCGAN generator to obtain a low resolution image
    4. run the Pix2Pix model to obtain a high resulution image
    
    Arguments:
        G {Generator} -- DCGAN generator
        pix2pix {networks.UnetGenerator} -- Low resolution to high resolution Pix2Pix model
        target_image {np.ndarray} -- The image to project
        E {Encoder} -- The DCGAN encoder
    
    Keyword Arguments:
        dcgan_img_size {int} -- Low resolution image size (default: {64})
        pix2pix_img_size {int} -- High resolution image size (default: {128})
    
    Returns:
        Tuple[np.ndarray, torch.Tensor] -- The projected high resolution image and the latent vector that was used to generate it.
    """    
    n_bins = 2.0 ** args.n_bits

    # target_image (3,64,64)
    tensor_image = torch.Tensor(target_image).cuda().reshape(1,3,dcgan_img_size,dcgan_img_size)
    tensor_image = tensor_image.type(dtype=torch.float32)
    tensor_image.retain_grad()
    tensor_image.requires_grad = True

    image = tensor_image - tensor_image.min()
    image = image / image.max() * 255

    if args.n_bits < 8:
        image = torch.floor(image / 2 ** (8 - args.n_bits))

    image = image / n_bins - 0.5

    log_p, logdet, z_outs = G(image / n_bins)
    # model(image + torch.rand_like(image) / n_bins)

    logdet = logdet.mean()

    loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)

    G.zero_grad()
    if tensor_image.grad is not None:
        tensor_image.grad.detach_()
        tensor_image.grad.zero_()

    loss.backward(retain_graph=True)
    eta = 1.0
    cur_grad = tensor_image.grad.cpu().numpy().copy()
    assert np.any(cur_grad!=0)
    pix_outputs = target_image + eta*cur_grad.reshape(3,-1,1)

    # return pix_outputs.detach().clamp(0,1).cpu().numpy().reshape(3,-1,1), z_outs
    return pix_outputs, z_outs

    # # reshape and normalize image
    # target_image = torch.Tensor(target_image).cuda().reshape(1,3,pix2pix_img_size,pix2pix_img_size)
    # target_image = F.interpolate(target_image, scale_factor=dcgan_img_size/pix2pix_img_size, mode='bilinear')
    # target_image = target_image.clamp(min=0)
    # target_image = target_image / target_image.max()
    # target_image = (target_image - 0.5) / 0.5

    # # Run dcgan
    # z = E(target_image)
    # dcgan_image = G(z)

    # # run pix2pix
    # pix_input = F.interpolate(dcgan_image, scale_factor=pix2pix_img_size/dcgan_img_size, mode='bilinear')
    # pix_outputs = pix2pix(pix_input)
    # out_image = utils.denorm(pix_outputs.detach()).clamp(0,1).cpu().numpy().reshape(3,-1,1)
    # return out_image, z

def morph_project_only(im1: np.ndarray, im2: np.ndarray, Generator: nn.Module, Generator_args: Namespace,
                       epsilon: float=20.0, L: int=9, dcgan_size: int=64,) -> Tuple:
    """Generates 3 morphing processes given two images. 
    The first is simple Wasserstein Barycenters, the second is our algorithm and 
    the third is a simple GAN latent space linear interpolation
    
    Arguments:
        im1 {np.ndarray} -- source image
        im2 {np.ndarray} -- destination image
        Generator {Generator} -- DCGAN generator (latent space to pixel space)
        Encoder {Encoder} -- DCGAN encoder (pixel space to latent space)
        pix2pix {networks.UnetGenerator} -- pix2pix model trained to increase an image resolution
    
    Keyword Arguments:
        epsilon {float} -- entropic regularization parameter (default: {20.0})
        L {int} -- number of images in the trasformation (default: {9})
        dcgan_size {int} -- DCGAN image size (low resolution) (default: {64})
        pix2pix_size {int} -- Pix2Pix image size (high resolution) (default: {128})
        simulation_name {str} -- name of the simulation. Affects the saved file names (default: {"image_interpolation"})
        results_path {str} -- the path to save the results in (default: {"results"})
    """
    # img_size = im1.shape[:2]
    # im1, im2 = (I.transpose(2,0,1).reshape(3,-1,1) for I in (im1, im2))
    img_size = im1.shape[1:]
    im1, im2 = (I.reshape(3,-1,1) for I in (im1, im2))

    print("Preparing transportation cost matrix...")
    C = generate_metric(img_size)
    Q = np.concatenate([im1, im2], axis=-1)
    Q, max_val, Q_counts = preprocess_Q(Q)
    out_ours = []
    out_GAN = []
    out_OT = []

    print("Computing transportation plan...")
    for dim in range(3):
        print(f"Color space {dim+1}/3")
        out_OT.append([])
        P = sinkhorn(Q[dim,:,0], Q[dim,:,1], C, img_size[0], img_size[1], epsilon)
        for t in tqdm(np.linspace(0,1,L)):
            out_OT[-1].append(max_val - generate_interpolation(img_size[0],img_size[1],P,t)*((1-t)*Q_counts[dim,0,0] + t*Q_counts[dim,0,1]))
    out_OT = [np.stack(im_channels, axis=0) for im_channels in zip(*out_OT)]
    
    print("Computing GAN projections...")
    # Project OT results on GAN
    GAN_projections = [project_on_generator(Generator, Generator_args, I, dcgan_img_size=dcgan_size) for I in out_OT]
    GAN_projections_images, GAN_projections_noises = zip(*GAN_projections)
    out_ours = GAN_projections_images

    # # Linearly interpolate GAN's latent space
    # noise1, noise2 = GAN_projections_noises[0].cuda(), GAN_projections_noises[-1].cuda()
    # for t in np.linspace(0,1,L):
    #     t = float(t)  # cast numpy object to primative type
    #     GAN_image = Generator((1-t)*noise1 + t*noise2)
    #     GAN_image = F.interpolate(GAN_image, scale_factor=2, mode='bilinear')
    #     pix_outputs = pix2pix(GAN_image)
    #     GAN_image = utils.denorm(pix_outputs.detach()).cpu().numpy().reshape(3,-1,1)
    #     out_GAN.append(GAN_image.clip(0,1))
    
    out_OT = torch.stack([torch.Tensor(im).reshape(3,*img_size) for im in out_OT])
    out_ours = torch.stack([torch.Tensor(im).reshape(3,*img_size) for im in out_ours])
    # out_GAN = torch.stack([torch.Tensor(im).reshape(3,*img_size) for im in out_GAN])

    return out_OT,out_ours,out_GAN
