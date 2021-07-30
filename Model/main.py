from option import args
from utils import mkExpDir
from dataset import dataloader
from trainer import Trainer
from model import model
import pandas as pd
import os
import torch
import torch.nn as nn
import warnings



warnings.filterwarnings('ignore')

if __name__ == '__main__':
    ###make save dir
    _logger = mkExpDir(args)
    ### dataset loader
    _dataloader = dataloader.get_dataloader(args)

    ### device setting
    device = torch.device('cpu' if args.cpu else 'cuda')
   
    ### model setting
    _model = model.VQAC(args).to(device)
    ### loss function setting
    answerset = pd.read_csv(args.answer_path, header=None)[0]
    ### trainer setting 
    t = Trainer(args,_logger,_dataloader,_model,answerset)
    ### training & validation 
    if args.test == True:
        acc=t.test()
    else:
        for epoch in range(args.now_epoch, args.num_epoch):
            t.train(current_epoch=epoch, is_init= args.is_init)
            


