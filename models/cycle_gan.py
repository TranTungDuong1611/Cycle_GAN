import os
import sys
import random

sys.path.append(os.getcwd())

import torch
from models.generator import *
from models.discriminator import *

# Image pool
class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


class CycleGAN(nn.Module):
    def __init__(self, opt, device):
        """ CycleGAN class

        Args:
            opt (argparse.Namespace): Options for CycleGAN
        """
        
        super(CycleGAN, self).__init__()
        
        self.opt = opt
        self.device = device
        self.fake_pool_A = ImagePool(pool_size=opt['model']['pool_size'])
        self.fake_pool_B = ImagePool(pool_size=opt['model']['pool_size'])
        
        # Initialize generators and discriminators
        self.netG_A = init_generator().to(self.device)
        self.netG_B = init_generator().to(self.device)
        
        # initialize discriminators when training
        if self.opt['isTrain']:
            self.netD_A = init_discriminator().to(self.device)
            self.netD_B = init_discriminator().to(self.device)
        
        # define loss functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        # define optimizers
        self.optimizer_G = torch.optim.Adam(
            list(self.netG_A.parameters()) + list(self.netG_B.parameters()),
            lr=opt['model']['lr'],
            betas=(opt['model']['beta1'], opt['model']['beta2'])
        )
        if self.opt['isTrain']:
            self.optimizer_D = torch.optim.Adam(
                list(self.netD_A.parameters()) + list(self.netD_B.parameters()),
                lr=['model']['lr'],
                betas=(opt['model']['beta1'], opt['model']['beta2'])
        )
        # define schedulers
        self.scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer_G,
            step_size=opt['model']['lr_step'],
            gamma=opt['model']['lr_gamma']
        )
        if self.opt['isTrain']:
            self.scheduler_D = torch.optim.lr_scheduler.StepLR(
                self.optimizer_D,
                step_size=opt['model']['lr_step'],
                gamma=opt['model']['lr_gamma']
        )

    def set_input(self, real_A, real_B):
        """ data of a batch for training
        params:
            real_A (torch.Tensor): (follow data in the dataloder)
            real_B (torch.Tensor): (follow data in the dataloder)
        """
        self.real_A = real_A
        self.real_B = real_B

        self.real_A = self.real_A.to(self.device)
        self.real_B = self.real_B.to(self.device)

    def set_requires_grad(self, nets, requires_grad=True):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
        
    def forward(self):
        """ Forward pass through the model
        Args:
            real_A (torch.Tensor): Input image from domain A
            real_B (torch.Tensor): Input image from domain B
        Returns:
            fake_B (torch.Tensor): Generated image in domain B
            rec_A (torch.Tensor): Reconstructed image in domain A
            fake_A (torch.Tensor): Generated image in domain A
            rec_B (torch.Tensor): Reconstructed image in domain B
        """
        # Generate fake images
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)
        
        return self.fake_B, self.rec_A, self.fake_A, self.rec_B
    
    def backward_D_A(self):
        """ Backward pass for the discriminators
        """
        # Compute GAN loss for discriminator A
        pred_real_A = self.netD_A(self.real_A)
        # get image from image pool
        fake_A = self.fake_pool_A.query(self.fake_A)
        # pass fake image into the discriminator
        pred_fake_A = self.netD_A(fake_A.detach())
        loss_D_A = (self.criterion_GAN(pred_real_A, torch.ones_like(pred_real_A)) + self.criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A)))*0.5
        
        # Backpropagation
        loss_D_A.backward()
        
        return loss_D_A
    
    def backward_D_B(self):
        """ Backward pass for the discriminators
        """
        # Compute GAN loss for discriminator B
        pred_real_B = self.netD_B(self.real_B)
        # get image from image pool
        fake_B = self.fake_pool_B.query(self.fake_B)
        # pass fake image into the discriminator
        pred_fake_B = self.netD_B(fake_B.detach())
        loss_D_B = (self.criterion_GAN(pred_real_B, torch.ones_like(pred_real_B)) + self.criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B)))*0.5
        
        # Backpropagation
        loss_D_B.backward()
        
        return loss_D_B
    
    def backward_G(self):
        """ Backward pass for the generators
        Args:
            real_A (torch.Tensor): Real image from domain A
            real_B (torch.Tensor): Real image from domain B
        """
        # Compute GAN loss for generator A
        pred_fake_A = self.netD_A(self.fake_A)
        loss_G_A = self.criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))
        
        # Compute GAN loss for generator B
        pred_fake_B = self.netD_B(self.fake_B)
        loss_G_B = self.criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))
        
        # Compute cycle consistency loss
        loss_cycle = self.criterion_cycle(self.rec_A, self.real_A) + self.criterion_cycle(self.rec_B, self.real_B)
        
        # Compute identity loss
        loss_identity = self.criterion_identity(self.netG_A(self.real_B), self.real_B) + self.criterion_identity(self.netG_B(self.real_A), self.real_A)
        
        # Backpropagation
        loss_G = (loss_G_A + loss_G_B) + self.opt['model']['lambda_cycle'] * loss_cycle + 5 * loss_identity
        loss_G.backward()
        
        return (loss_G_A + loss_G_B), self.opt['model']['lambda_cycle'] * loss_cycle, 5 * loss_identity
    
    def optimize_parameters(self):
        # forward pass
        self.forward()

        # Train generator
        self.set_requires_grad([self.netD_A, self.netD_B], requires_grad=False) # Freeze D when training G
        self.optimizer_G.zero_grad()
        loss_G_gan, loss_G_cycle, loss_G_idt = self.backward_G()
        self.optimizer_G.step()
        
        # Train discriminator
        self.set_requires_grad([self.netD_A, self.netD_B], requires_grad=True) # Unfreeze D for training D
        self.optimizer_D.zero_grad()
        loss_D_A = self.backward_D_A()
        loss_D_B = self.backward_D_B()
        total_loss_D = (loss_D_A + loss_D_B)
        self.optimizer_D.step()

        return loss_G_gan, loss_G_cycle, loss_G_idt, total_loss_D