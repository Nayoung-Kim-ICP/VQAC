from loss import discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pdb


def get_loss_dict(args, logger):
    loss = {}
    if (abs(args.rec_w - 0) <= 1e-8):
        raise SystemExit('NotImplementError: ReconstructionLoss must exist!')
    else:
        loss['rec_loss'] = 
    
    return loss

def get_loss_dict1():
    loss = {}

    loss['per_loss'] = PerceptualLoss()
    
    return loss
