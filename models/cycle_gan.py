import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from tqdm import tqdm
from torchsummary import summary
from models.generator import *
from models.discriminator import *

class CycleGAN(nn.Module):
    def __init__(self, opt, device):
        """ CycleGAN class

        Args:
            opt (argparse.Namespace): Options for CycleGAN
        """
        
        super(CycleGAN, self).__init__()
        
        self.opt = opt
        self.device = device
        
        # Initialize generators and discriminators
        self.netG_A = init_generator(opt['model']['image_size'], in_channels=3, down_factor=opt['model']['down_factor']).to(self.device)
        self.netG_B = init_generator(opt['model']['image_size'], in_channels=3, down_factor=opt['model']['down_factor']).to(self.device)
        
        # initialize discriminators when training
        if self.opt['isTrain']:
            self.netD_A = init_discriminator(in_channels=3).to(self.device)
            self.netD_B = init_discriminator(in_channels=3).to(self.device)
        
        # define loss functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        # define optimizers
        self.optimizer_G = torch.optim.Adam(
            list(self.netG_A.parameters()) + list(self.netG_B.parameters()),
            lr=opt['model']['lr'],
            betas=(opt['model']['beta1'], 0.999)
        )
        if self.opt['isTrain']:
            self.optimizer_D = torch.optim.Adam(
                list(self.netD_A.parameters()) + list(self.netD_B.parameters()),
                lr=opt['model']['lr'],
                betas=(opt['model']['beta1'], 0.999)
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
        
    def forward(self, real_A, real_B):
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
        fake_B = self.netG_A(real_A)
        rec_A = self.netG_B(fake_B)
        fake_A = self.netG_B(real_B)
        rec_B = self.netG_A(fake_A)
        
        return fake_B, rec_A, fake_A, rec_B
    
    def backward_D_A(self, real_A, fake_A):
        """ Backward pass for the discriminators
        Args:
            real_A (torch.Tensor): Real image from domain A
            real_B (torch.Tensor): Real image from domain B
            fake_A (torch.Tensor): Fake image generated in domain A
            fake_B (torch.Tensor): Fake image generated in domain B
        """
        # Compute GAN loss for discriminator A
        pred_real_A = self.netD_A(real_A)
        pred_fake_A = self.netD_A(fake_A.detach())
        loss_D_A = self.criterion_GAN(pred_real_A, torch.ones_like(pred_real_A)) + self.criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))
        
        # Backpropagation
        loss_D_A.backward()
        
        return loss_D_A
    
    def backward_D_B(self, real_B, fake_B):
        """ Backward pass for the discriminators
        Args:
            real_A (torch.Tensor): Real image from domain A
            real_B (torch.Tensor): Real image from domain B
            fake_A (torch.Tensor): Fake image generated in domain A
            fake_B (torch.Tensor): Fake image generated in domain B
        """
        # Compute GAN loss for discriminator B
        pred_real_B = self.netD_B(real_B)
        pred_fake_B = self.netD_B(fake_B.detach())
        loss_D_B = self.criterion_GAN(pred_real_B, torch.ones_like(pred_real_B)) + self.criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))
        
        # Backpropagation
        loss_D_B.backward()
        
        return loss_D_B
    
    def backward_G(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B):
        """ Backward pass for the generators
        Args:
            real_A (torch.Tensor): Real image from domain A
            real_B (torch.Tensor): Real image from domain B
            fake_A (torch.Tensor): Fake image generated in domain A
            fake_B (torch.Tensor): Fake image generated in domain B
            rec_A (torch.Tensor): Reconstructed image in domain A
            rec_B (torch.Tensor): Reconstructed image in domain B
        """
        # Compute GAN loss for generator A
        pred_fake_A = self.netD_A(fake_A)
        loss_G_A = self.criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))
        
        # Compute GAN loss for generator B
        pred_fake_B = self.netD_B(fake_B)
        loss_G_B = self.criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))
        
        # Compute cycle consistency loss
        loss_cycle = self.criterion_cycle(rec_A, real_A) + self.criterion_cycle(rec_B, real_B)
        
        # Compute identity loss
        loss_identity = self.criterion_identity(self.netG_A(real_B), real_B) + self.criterion_identity(self.netG_B(real_A), real_A)
        
        # Backpropagation
        loss_G = (loss_G_A + loss_G_B) * 0.5 + self.opt['model']['lambda_cycle'] * loss_cycle + self.opt['model']['lambda_identity'] * loss_identity
        loss_G.backward()
        
        return loss_G
    
    def train(self, dataloader):
        """ Set the model to training mode """
        self.netG_A.train()
        self.netG_B.train()
        
        if self.opt['isTrain']:
            self.netD_A.train()
            self.netD_B.train()
            
        for epoch in range(self.opt['model']['epochs']):
            total_loss_G = 0.0
            total_loss_D = 0.0
            print('hello')
            for real_A, real_B in tqdm(dataloader):
                real_A = real_A.to(self.device)
                real_B = real_B.to(self.device)
                
                # Forward pass
                fake_B, rec_A, fake_A, rec_B = self(real_A, real_B)
                
                # Train generator
                self.optimizer_G.zero_grad()
                loss_G = self.backward_G(real_A, real_B, fake_A, fake_B, rec_A, rec_B)
                self.optimizer_G.step()
                total_loss_G += loss_G.item()
                
                # Train discriminator
                self.optimizer_D.zero_grad()
                loss_D_A = self.backward_D_A(real_A, fake_A)
                loss_D_B = self.backward_D_B(real_B, fake_B)
                total_loss_D = (loss_D_A + loss_D_B) * 0.5
                self.optimizer_D.step()
                
            # Print loss
            print(f"Epoch [{epoch}/{self.opt['model']['epochs']}], Loss G: {total_loss_G/len(dataloader)}, Loss D: {total_loss_D/len(dataloader)}")
