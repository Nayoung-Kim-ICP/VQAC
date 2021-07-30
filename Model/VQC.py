import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np



class ResidueWeightedvectorGeneration(nn.Module):
    def __init__(self,input_size,input_number,hidden_size):
        super(ResidueWeightedvectorGeneration, self).__init__()
        
        self.RWvec = ResidueWeightedvector(hidden_size)
        self.Decoder = Decoder(hidden_size,1000)


    def forward(self, apperances,apperances_next,residue,text):
        apperance_batch = []
        motion_batch = []
        batch_size = apperances.shape[0]
        Fusion=[]
        for j in range(batch_size):
            motions,alpha = self.RWvec(apperances[j,:], apperances_next[j,:], residue[j,:])
            Fusion.append(torch.sum(motion)+torch.sum(text))
            output_s = torch.cat(Fusion, dim=0)

        return torch.cat(apperance_batch, dim=0),torch.cat(motion_batch, dim=0),alpha



class ResidueWeightedvector(nn.Module):
    def __init__(self, hidden_size,input_number):
        super(ResidueWeightedvector, self).__init__()

        self.alpha_layer = nn.Linear(input_number,1)
        self.th = 1
        self.FC_m1 = nn.Linear(7*7*512,hidden_size*2)
        self.FC_m2 = nn.Linear(hidden_size*2,hidden_size)


    def alpha(self,x):
        output = torch.sigmoid(self.alpha_layer(x))
        return output


    def forward(self, apperances, apperances_next, Residue):
        
        alpha_t = self.alpha(Residue)
        
        alpha_t=alpha_t.view(alpha_t.size(0),1,1,1)

        input_fc = (1-alpha_t)*apperances+(alpha_t)*apperances_next
        input_fc = input_fc.view(input_fc.size(0),input_fc.size(1)*input_fc.size(2)*input_fc.size(3))

        output_motion = self.FC_m2(F.relu(self.FC_m1(input_fc)))
        return output_motion.unsqueeze(0),alpha_t


class Decoder(nn.Module):
    def __init__(self,hidden_size,answer_size):
        super(Decoder, self).__init__()
        self.FC1 = nn.Linear(hidden_size,hidden_size)
        self.FC2 = nn.Linear(hidden_size,answer_size) 

    def forward(self, multimodal_input):
     
        outputs = F.relu(self.FC1(multimodal_input))
        outputs = self.FC2(outputs)
        _,mx_idx = torch.max(output,1)
        
        return outputs, mx_idx




