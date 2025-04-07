import os
import sys
import json
sys.path.append(os.getcwd())
import argparse
import torch
from models.cycle_gan import CycleGAN
from data.dataloader import *

def main():
    parcer = argparse.ArgumentParser()
    parcer.add_argument('--isTrain', action='store_true', help='is training or testing')
    
    args = parcer.parse_args()
    
    # load config file
    with open('config.json') as f:
        opt = json.load(f)
        
    if args.isTrain:
        opt['isTrain'] = True
    else:
        opt['isTrain'] = False
        
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # define CycleGAN model
    model = CycleGAN(opt, device)
    model.to(device)
    
    if opt['isTrain']:
        # Load training data
        train_loader = get_dataloader(mode='train')
        
        model.train(train_loader)
        
if __name__ == "__main__":
    main()