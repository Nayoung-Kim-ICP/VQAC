import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_module_video import *
from model.model_module_question import *
from model.attention_module import *

class VQAC(nn.Module):

    def __init__(self,args):
        """Set the hyper-parameters and build the modules."""
        super(VQAC, self).__init__()
        
        hidden_size = args.hidden_size
        vocab_size = args.vocab_size
        input_size = args.input_size
        input_number = args.input_number
        answer_size = args.answer_size
        text_embed_size = args.text_embed_size
        word_matrix_path = args.word_matrix_path
        vid_fusion_mode = args.vid_fusion_mode
        self.model_mode = args.model_mode
        word_matrix =  np.load(word_matrix_path)
        embed = nn.Embedding(vocab_size, text_embed_size)
        embed.weight.data.copy_(torch.from_numpy(word_matrix))

        self.QFM = QuestionEncoderLSTM(hidden_size,text_embed_size,embed)
        self.QGA = QuestionguidedAtt(input_size,hidden_size)
        self.MFG = motionFeature(input_size)
        self.RWG =  ResidueWeightedvectorG(input_size,input_number,hidden_size,answer_size,vid_fusion_mode)

        if self.model_mode == 'HME':
            self.HME_M = HME_M(hidden_size, embed)
            self.HME_D = HME_D(hidden_size, answer_size)
                 

    def forward(self, f_t, f_t_1, residue, qa, qa_len):

        E_q, batch_state = self.QFM(qa,qa_len)
        f_m = self.MFG(f_t,f_t_1)
        f_w = batch_state[1]

        if self.model_mode == 'HME':
            batch_a,batch_m, batch_smq,f_w = self.HME_M(f_t,f_m,qa,qa_len,batch_state)
            
        G_s,G_m = self.QGA(f_t,f_m,E_q)
        decoder_feature,outputs,predictions  = self.RWG(G_s,G_m,residue,f_w)

        if self.model_mode == 'HME':
            outputs, predictions = self.HME_D(batch_a,batch_m, batch_smq,decoder_feature) 
        return outputs,predictions



