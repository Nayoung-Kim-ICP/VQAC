import os
import pandas as pd
import h5py
import numpy as np
import random
import torch
from os import listdir
from os.path import join
from torch.utils.data import Dataset
from torchvision import transforms

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class ToTensor(object):
    def __call__(self, sample):
        questions, answers = sample['questions'], sample['answers']
        f_t, f_t_1, r_t  = sample['f_t'], sample['f_t_1'], sample['r_t']
        f_t = np.array(f_t).astype(np.float32)
        f_t_1 = np.array(f_t_1).astype(np.float32)
        r_t = np.array(r_t).astype(np.float32)
        f_t = torch.from_numpy(f_t)
        f_t_1 = torch.from_numpy(f_t_1)
        r_t = torch.from_numpy(r_t)
       
        questions = np.array(questions).astype(np.int64)
        answers = np.array(answers).astype(np.int64)
        answers = torch.from_numpy(answers)

        return {'f_t': f_t,
                'f_t_1': f_t_1,
                'r_t': r_t,
                'questions':questions,
                'answers': answers
                }


class ToTensorTest(object):
    def __call__(self, sample):
        questions = sample['questions']
        f_t, f_t_1, r_t  = sample['f_t'], sample['f_t_1'], sample['r_t']
        f_t = np.array(f_t).astype(np.float32)
        f_t_1 = np.array(f_t_1).astype(np.float32)
        r_t = np.array(r_t).astype(np.float32)
        f_t = torch.from_numpy(f_t)
        f_t_1 = torch.from_numpy(f_t_1)
        r_t = torch.from_numpy(r_t)
       
        questions = np.array(questions).astype(np.int64)

        return {'f_t': f_t,
                'f_t_1': f_t_1,
                'r_t': r_t,
                'question':questions
                }


class TrainSet(Dataset):
    def __init__(self, args, transform=ToTensor() ):
                
        vid_name = './video_data/resnet152_train.h5'
        res_name = './video_data/resi_train.h5'

        self.vid_feature = h5py.File(os.path.join(args.data_path, vid_name),'r')
        self.res_feature = h5py.File(os.path.join(args.data_path, res_name),'r')

        self.qa_ = pd.read_json(
            os.path.join(args.qa_path, './text_data/train_qa_encode.json'))

        self.qa_['question_length'] = self.qa_.apply(
            lambda row: len(row['question'].split()), axis=1)
        self.max_len_qa = args.max_len_qa
        self.transform = transform
       
    def __len__(self):
        return len(self.qa_)

    def __getitem__(self, idx):
        ### HR
        qa = self.qa_['question_encode'][idx]
        ans = self.qa_['answer_encode'][idx]
        v_id = self.qa_['video_id'][idx]
        
        qid = [int(x) for x in qa.split(',')]
        question_lengths = len(qid)
        if question_lengths ==0:
            question_lengths = 1
        if question_lengths > self.max_len_qa:
            qid = qid[:self.max_len_qa]
            question_lengths = self.max_len_qa
        else:
            qid = np.pad(qid, (0, self.max_len_qa - len(qid)), 'constant')
        
        vid_name =str(v_id)
        r_t = np.reshape(self.res_feature[vid_name],(20,49))
        f_t = self.vid_feature[vid_name][0,:,:,:,:]
        f_t_1 = self.vid_feature[vid_name][1,:,:,:,:]
        sample = {'f_t': f_t,  
                  'f_t_1': f_t_1,
                  'r_t': r_t,
                  'questions':qid,
                  'answers':ans}

        
        sample = self.transform(sample)
        sample['question_lengths']=question_lengths
        return sample


class ValSet(Dataset):
    def __init__(self, args, transform=ToTensorTest() ):
        vid_name = './video_data/resnet152_val.h5'
        res_name = './video_data/resi_val.h5'

        self.vid_feature = h5py.File(os.path.join(args.data_path, vid_name),'r')
        self.res_feature = h5py.File(os.path.join(args.data_path, res_name),'r')

        self.qa_ = pd.read_json(
            os.path.join(args.qa_path, './text_data/val_qa_encode.json'))

        self.qa_['question_length'] = self.qa_.apply(
            lambda row: len(row['question'].split()), axis=1)

        self.transform = transform
        
    def __len__(self):
        return len(self.qa_)

    def __getitem__(self, idx):

        qa = self.qa_['question_encode'][idx]#.values
        ans = self.qa_['answer'][idx]#.values
        v_id = self.qa_['video_id'][idx]#.values

                
        qid = [int(x) for x in qa.split(',')]
        question_lengths = len(qid)
        
        
        vid_name =str(v_id)
        r_t = np.reshape(self.res_feature[vid_name],(20,49))
        f_t = self.vid_feature[vid_name][0,:,:,:,:]
        f_t_1 = self.vid_feature[vid_name][1,:,:,:,:]
        sample = {'f_t': f_t,  
                  'f_t_1': f_t_1,
                  'r_t': r_t,
                  'questions':qid,
                  'answer':ans}


        sample = self.transform(sample)
        sample['question_lenght']=question_lengths
        sample['test_example_total']=len(self.qa_)
        return sample


class TestSet(Dataset):
    def __init__(self, args, transform=ToTensorTest() ):
        vid_name = './video_data/resnet152_test.h5'
        res_name = './video_data/resi_test.h5'

        self.vid_feature = h5py.File(os.path.join(args.data_path, vid_name),'r')
        self.res_feature = h5py.File(os.path.join(args.data_path, res_name),'r')

        self.qa_ = pd.read_json(
            os.path.join(args.qa_path, './text_data/test_qa_encode.json'))

        self.qa_['question_length'] = self.qa_.apply(
            lambda row: len(row['question'].split()), axis=1)
        self.transform = transform
       

    def __len__(self):
        return len(self.qa_)

    def __getitem__(self, idx):
        ### HR
        qa = self.qa_['question_encode'][idx]
        ans = self.qa_['answer'][idx]
        v_id = self.qa_['video_id'][idx]
        qid = [int(x) for x in qa.split(',')]
        question_lengths = len(qid)
        qa_text = self.qa_['question'][idx]
        vid_name =str(v_id)
        r_t = np.reshape(self.res_feature[vid_name],(20,49))
        f_t = self.vid_feature[vid_name][0,:,:,:,:]
        f_t_1 = self.vid_feature[vid_name][1,:,:,:,:]
        
        
        sample = {'f_t': f_t,  
                  'f_t_1': f_t_1,
                  'r_t': r_t,
                  'questions':qid}

        sample = self.transform(sample)
        return sample,question_lengths,v_id,ans,qa_text,len(self.qa_)

