from torch.utils.data import DataLoader
from importlib import import_module


def get_dataloader(args):
    ### import module###
    m = import_module('dataset.' +args.data_name)

    data_train = getattr(m, 'TrainSet')(args)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        
    data_test = getattr(m, 'TestSet')(args)
    dataloader_test = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=args.num_workers)

    data_val = getattr(m, 'ValSet')(args)
    dataloader_val = DataLoader(data_val, batch_size=1, shuffle=False, num_workers=args.num_workers)
    dataloader = {'train': dataloader_train, 'test': dataloader_test, 'val': dataloader_val}

    return dataloader
