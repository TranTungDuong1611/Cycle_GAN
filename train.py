import os
import sys
import json
sys.path.append(os.getcwd())

import torch
import argparse

from models.cycle_gan import *
from data.dataloader import *
from utils.util import *
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='config.json', help='path to your config file')
    args = parser.parse_args()
    
    # load config file
    with open(f"{args.config_file}") as f:
        config = json.load(f)
        
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # define CycleGAN model
    cycleGan = CycleGAN(config, device)
    cycleGan.to(device)
    
    # Load training data
    train_loader = get_dataloader(mode='train')
    test_loader = get_dataloader(mode='test')
    
    # make checkpoint dir if not exist
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    # training loop
    for epoch in tqdm(range(config['model']['epochs'])):
        total_loss_G_gan = 0.0
        total_loss_G_cycle = 0.0
        total_loss_G_idt = 0.0
        total_loss_D = 0.0
        for real_A, real_B in train_loader:
            cycleGan.set_input(real_A, real_B)
            loss_G_gan, loss_G_cycle, loss_G_idt, loss_D = cycleGan.optimize_parameters()
            total_loss_G_gan += loss_G_gan
            total_loss_G_cycle += loss_G_cycle
            total_loss_G_idt += loss_G_idt
            total_loss_D += loss_D

        if (epoch+1) % 10 == 0 or epoch == 0:
            # Print loss
            print(f"Epoch [{epoch+1}/{config['model']['epochs']}], Loss G gan: {loss_G_gan/len(train_loader):.4f}, Loss G cycle: {loss_G_cycle/len(train_loader):.4f}, Loss G idt: {loss_G_idt/len(train_loader):.4f}, Loss D: {total_loss_D/len(train_loader):.4f}")
            # save output
            torch.save(cycleGan.state_dict(), 'checkpoints/weights.pt')
            # visualize the results
            with torch.no_grad():
                for (real_A, real_B) in test_loader:
                    real_A = real_A.to(device)
                    real_B = real_B.to(device)
                    cycleGan.set_input(real_A, real_B)
                    fake_B, rec_A, fake_A, rec_B = cycleGan()
                    visualize_results(real_A, fake_A, rec_A, real_B, fake_B, rec_B)
                    break
    
if __name__ == '__main__':
    main()