###########################################
# The original code is from 
#@inproceedings{fan-CVPR-2019,
#    author    = {Chenyou Fan, Xiaofan Zhang, Shu Zhang, Wensheng Wang, Chi Zhang, Heng Huang},
#    title     = "{Heterogeneous Memory Enhanced Multimodal Attention Model for Video Question Answering}"
#    booktitle = {CVPR},
#    year      = 2019
#} [[link]](https://arxiv.org/pdf/1904.04357.pdf)
# We modify this code for our implementation
############################################


import torch
import torch.nn as nn
import torch.nn.functional as F
from model.memory_hme import MemoryRamModule,MemoryRamTwoStreamModule

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)

class TemporalAttentionModule(nn.Module):

    def __init__(self, input_size, hidden_size=512):
        """Set the hyper-parameters and build the layers."""
        super(TemporalAttentionModule, self).__init__()
        self.input_size = input_size   # in most cases, 2*hidden_size
        self.hidden_size = hidden_size

        self.Wa = nn.Parameter(torch.FloatTensor(input_size, hidden_size),requires_grad=True)
        self.Ua = nn.Parameter(torch.FloatTensor(input_size, hidden_size),requires_grad=True)
        self.Va = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.ba = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)
        
        self.init_weights()
        
        
    def init_weights(self):
        self.Wa.data.normal_(0.0, 0.01)
        self.Ua.data.normal_(0.0, 0.01)
        self.Va.data.normal_(0.0, 0.01)
        self.ba.data.fill_(0)


    def forward(self, hidden_frames, hidden_text, inv_attention=False):

        Uh = torch.matmul(hidden_text, self.Ua)  # (1,512)
        Uh = Uh.view(Uh.size(0),1,Uh.size(1)) # (1,1,512)

        # see appendices A.1.2 of neural machine translation
        # Page 12 last line
        Ws = torch.matmul(hidden_frames, self.Wa) # (1,T,512)
        #print('Temporal Ws size',Ws.size())       # (1, T, 512)
        att_vec = torch.matmul( torch.tanh(Ws + Uh + self.ba), self.Va )
        
        if inv_attention==True:
            att_vec = - att_vec
        
        att_vec = F.softmax(att_vec, dim=1) # normalize by Softmax, see Eq(15)
        att_vec = att_vec.view(att_vec.size(0),att_vec.size(1),1) # expand att_vec from 1xT to 1xTx1 

        # Hori ICCV 2017
        # Eq(10) c_i
        ht_weighted = att_vec * hidden_frames
        ht_sum = torch.sum(ht_weighted, dim=1)
        return ht_sum

class MultiModalAttentionModule(nn.Module):

    def __init__(self, hidden_size=512, simple=False):
        """Set the hyper-parameters and build the layers."""
        super(MultiModalAttentionModule, self).__init__()

        self.hidden_size = hidden_size
        self.simple=simple
        
        # alignment model
        # see appendices A.1.2 of neural machine translation
        
        self.Wav = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Wat = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Uav = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Uat = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Vav = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.Vat = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.bav = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)
        self.bat = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)

        self.Whh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Wvh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Wth = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.bh = nn.Parameter(torch.FloatTensor(1,1,hidden_size),requires_grad=True)

        self.video_sum_encoder = nn.Linear(hidden_size, hidden_size) 
        self.question_sum_encoder = nn.Linear(hidden_size, hidden_size) 

        self.Wb = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Vbv = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Vbt = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.bbv = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.bbt = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.wb = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)
        self.init_weights()
        
    def init_weights(self):

        self.Wav.data.normal_(0.0, 0.1)
        self.Wat.data.normal_(0.0, 0.1)
        self.Uav.data.normal_(0.0, 0.1)
        self.Uat.data.normal_(0.0, 0.1)
        self.Vav.data.normal_(0.0, 0.1)
        self.Vat.data.normal_(0.0, 0.1)
        self.bav.data.fill_(0)
        self.bat.data.fill_(0)

        self.Whh.data.normal_(0.0, 0.1)
        self.Wvh.data.normal_(0.0, 0.1)
        self.Wth.data.normal_(0.0, 0.1)
        self.bh.data.fill_(0)

        self.Wb.data.normal_(0.0, 0.01)
        self.Vbv.data.normal_(0.0, 0.01)
        self.Vbt.data.normal_(0.0, 0.01)
        self.wb.data.normal_(0.0, 0.01)
        
        self.bbv.data.fill_(0)
        self.bbt.data.fill_(0)

        
    def forward(self, h, hidden_frames, hidden_text, inv_attention=False):
        
        #print self.Uav
        # hidden_text:  1 x T1 x 1024 (looks like a two layer one-directional LSTM, combining each layer's hidden)
        # hidden_frame: 1 x T2 x 1024 (from video encoder output, 1024 is similar from above)

        #print hidden_frames.size(),hidden_text.size()
        Uhv = torch.matmul(h, self.Uav)  # (1,512)
        Uhv = Uhv.view(Uhv.size(0),1,Uhv.size(1)) # (1,1,512)

        Uht = torch.matmul(h, self.Uat)  # (1,512)
        Uht = Uht.view(Uht.size(0),1,Uht.size(1)) # (1,1,512)
        
        #print Uhv.size(),Uht.size()
        
        Wsv = torch.matmul(hidden_frames, self.Wav) # (1,T,512)
        #print Wsv.size()
        att_vec_v = torch.matmul( torch.tanh(Wsv + Uhv + self.bav), self.Vav )
        
        Wst = torch.matmul(hidden_text, self.Wat) # (1,T,512)
        att_vec_t = torch.matmul( torch.tanh(Wst + Uht + self.bat), self.Vat )
        
        if inv_attention==True:
            att_vec_v = -att_vec_v
            att_vec_t = -att_vec_t
            

        att_vec_v = torch.softmax(att_vec_v, dim=1)
        att_vec_t = torch.softmax(att_vec_t, dim=1)

        
        att_vec_v = att_vec_v.view(att_vec_v.size(0),att_vec_v.size(1),1) # expand att_vec from 1xT to 1xTx1 
        att_vec_t = att_vec_t.view(att_vec_t.size(0),att_vec_t.size(1),1) # expand att_vec from 1xT to 1xTx1 
                
        hv_weighted = att_vec_v * hidden_frames
        hv_sum = torch.sum(hv_weighted, dim=1)
        hv_sum2 = self.video_sum_encoder(hv_sum)

        ht_weighted = att_vec_t * hidden_text
        ht_sum = torch.sum(ht_weighted, dim=1)
        ht_sum2 = self.question_sum_encoder(ht_sum)        
        
        
        Wbs = torch.matmul(h, self.Wb)
        mt1 = torch.matmul(ht_sum, self.Vbt) + self.bbt + Wbs
        mv1 = torch.matmul(hv_sum, self.Vbv) + self.bbv + Wbs
        mtv =  torch.tanh(torch.cat([mv1,mt1],dim=0))
        mtv2 = torch.matmul(mtv, self.wb)
        beta = torch.softmax(mtv2,dim=0)

        output = torch.tanh( torch.matmul(h,self.Whh) + beta[0] * hv_sum2 + 
                             beta[1] * ht_sum2 + self.bh )
        output = output.view(output.size(1),output.size(2))
        
        return output

class HME_M(nn.Module):

    def __init__(self, hidden_size, embed, max_len=20, dropout=0.2, iter_num=3):

        """Set the hyper-parameters and build the layers."""
        super(HME_M, self).__init__()
                
        ### inital setting ###
        text_embed_size = 300 # should be 300
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.iter_num = iter_num
        self.embed = embed

        ### temporal attention modules ###
        self.TpAtt_a = TemporalAttentionModule(hidden_size*2, hidden_size)
        self.TpAtt_m = TemporalAttentionModule(hidden_size*2, hidden_size)
   
        ### text & video encoding modules ###  
        self.comp_vid = torch.nn.Sequential()
        self.comp_vid.add_module('conv', conv3x3(hidden_size,4)) 
        self.comp_vid.add_module('relu', nn.ReLU(inplace=True))  
        self.fully_vid = nn.Linear(28*28*4,hidden_size) 
               
        self.lstm_text_1 = nn.LSTMCell(text_embed_size, hidden_size)
        self.lstm_text_2 = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm_video_1a = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm_video_2a = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm_video_1m = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm_video_2m = nn.LSTMCell(hidden_size, hidden_size)

        ### fusion modules ###
        self.lstm_mm_1 = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm_mm_2 = nn.LSTMCell(hidden_size, hidden_size)
        self.hidden_encoder_1 = nn.Linear(hidden_size * 2, hidden_size) 
        self.hidden_encoder_2 = nn.Linear(hidden_size * 2, hidden_size) 
        self.mm_att = MultiModalAttentionModule(hidden_size)
        self.drop_keep_prob_final_att_vec = nn.Dropout(dropout)

        ### memory modules ###
        self.mrm_vid = MemoryRamTwoStreamModule(hidden_size, hidden_size, max_len)
        self.mrm_txt = MemoryRamModule(hidden_size, hidden_size, max_len)
        self.init_weights()

    def init_weights(self, ):
        """Initialize weights."""
        self.mrm_vid.init_weights()
        self.mrm_txt.init_weights()

    def init_hiddens(self):
        s_t = torch.zeros(1, self.hidden_size).cuda()
        s_t2 = torch.zeros(1, self.hidden_size).cuda()
        c_t = torch.zeros(1, self.hidden_size).cuda()
        c_t2 = torch.zeros(1, self.hidden_size).cuda()
        return s_t,s_t2,c_t,c_t2
    
    
    def mm_module_v1(self,svt_tmp,memory_ram_vid,memory_ram_txt,loop=3):
    
        sm_q1,sm_q2,cm_q1,cm_q2 = self.init_hiddens()
        mm_oo = self.drop_keep_prob_final_att_vec(torch.tanh(self.hidden_encoder_1(svt_tmp)))
        
        for _ in range(loop):
        
            sm_q1, cm_q1 = self.lstm_mm_1(mm_oo, (sm_q1, cm_q1))
            sm_q2, cm_q2 = self.lstm_mm_2(sm_q1, (sm_q2, cm_q2))
        
            mm_o1 = self.mm_att(sm_q2,memory_ram_vid,memory_ram_txt)
            mm_o2 = torch.cat((sm_q2,mm_o1),dim=1)
            mm_oo = self.drop_keep_prob_final_att_vec(torch.tanh(self.hidden_encoder_2(mm_o2)))
        
        smq = torch.cat( (sm_q1,sm_q2), dim=1)

        return smq
    

    def forward(self, f_t,f_m,questions,question_lengths,batch_state):
       
        batch_vid_a = []
        batch_vid_m = []
        batch_smq = []
        batch_fw = []
        bsize = len(questions)
        batch_size = len(questions)  
        features_questions = self.embed(questions)
        
        nImg = f_t.size(1) 
        s1_t1,s1_t2,c1_t1,c1_t2 = batch_state[0], batch_state[1], batch_state[2], batch_state[3]
        for j in range(batch_size):

            nQuestionWords = question_lengths[j]
            c_f_m = self.comp_vid(f_m[j])
            motion_feature = self.fully_vid(c_f_m.view(c_f_m.size(0),c_f_m.size(1)*c_f_m.size(2)*c_f_m.size(3)))
            c_f_s = self.comp_vid(f_t[j])
            appear_feature = self.fully_vid(c_f_s.view(c_f_s.size(0),c_f_s.size(1)*c_f_s.size(2)*c_f_s.size(3)))

            ###########################################             
            # run video encoder with spatial attention
            ###########################################
            sV_t1a,sV_t2a,cV_t1a,cV_t2a = s1_t1[j],s1_t2[j],c1_t1[j],c1_t2[j]
            sV_t1m,sV_t2m,cV_t1m,cV_t2m = s1_t1[j],s1_t2[j],c1_t1[j],c1_t2[j]

            # record each time t, hidden states, for later temporal attention after text encoding
            hidden_array_1a = []
            hidden_array_2a = []
            hidden_array_1m = []
            hidden_array_2m = []
            
            for i in range(nImg):

                
                feat_att_m = motion_feature[i:i+1,:]
                feat_att_a = appear_feature[i:i+1,:]
                
                sV_t1m, cV_t1m = self.lstm_video_1m(feat_att_m, (sV_t1m, cV_t1m))
                sV_t2m, cV_t2m = self.lstm_video_2m(sV_t1m, (sV_t2m, cV_t2m))

                sV_t1a, cV_t1a = self.lstm_video_1a(feat_att_a, (sV_t1a, cV_t1a))
                sV_t2a, cV_t2a = self.lstm_video_2a(sV_t1a, (sV_t2a, cV_t2a))
                
                sV_t1a_vec = sV_t1a.view(sV_t1a.size(0),1,sV_t1a.size(1))
                sV_t2a_vec = sV_t2a.view(sV_t2a.size(0),1,sV_t2a.size(1))
            
                hidden_array_1a.append(sV_t1a_vec)
                hidden_array_2a.append(sV_t2a_vec)
            
                sV_t1m_vec = sV_t1m.view(sV_t1m.size(0),1,sV_t1m.size(1))
                sV_t2m_vec = sV_t2m.view(sV_t2m.size(0),1,sV_t2m.size(1))
            
                hidden_array_1m.append(sV_t1m_vec)
                hidden_array_2m.append(sV_t2m_vec)
                

            sV_l1a = torch.cat(hidden_array_1a, dim=1)
            sV_l2a = torch.cat(hidden_array_2a, dim=1)
            sV_l1m = torch.cat(hidden_array_1m, dim=1)
            sV_l2m = torch.cat(hidden_array_2m, dim=1)
        
            sV_lla = torch.cat((sV_l1a,sV_l2a), dim=2)
            sV_llm = torch.cat((sV_l1m,sV_l2m), dim=2)

               
            #############################             
            # run text encoder second time
            #############################
            sT_t1,sT_t2,cT_t1,cT_t2 = self.init_hiddens()
            sT_t1,sT_t2 = sV_t1a+sV_t1m, sV_t2a+sV_t2m
            
            hidden_array_3 = []
            
                
            for i in range(nQuestionWords):
                input_question = features_questions[j,i:i+1]
                sT_t1, cT_t1 = self.lstm_text_1(input_question, (sT_t1, cT_t1))
                sT_t2, cT_t2 = self.lstm_text_2(sT_t1, (sT_t2, cT_t2))
                hidden_array_3.append(sT_t2)

            # here sT_t1, sT_t2 is the last hidden
            sT_t = torch.cat( (sT_t1,sT_t2), dim=1)  # should be of size (1,1024)
            
            
            #####################
            # temporal attention
            #####################
            vid_att_a = self.TpAtt_a(sV_lla, sT_t)
            vid_att_m = self.TpAtt_m(sV_llm, sT_t)


            ################
            # ram memory
            ################
            sT_rl = torch.cat(hidden_array_3, dim=0)

            memory_ram_vid = self.mrm_vid(sV_l2a[0,:,:], sV_l2m[0,:,:], nImg)
            memory_ram_txt = self.mrm_txt(sT_rl, nQuestionWords)
                
            svt_tmp = torch.cat((sV_t2a,sV_t2m),dim=1)
            smq = self.mm_module_v1(svt_tmp,memory_ram_vid,memory_ram_txt,self.iter_num)

            batch_vid_a.append(vid_att_a)
            batch_vid_m.append(vid_att_m)
            batch_smq.append(smq)
            batch_fw.append(torch.sum(memory_ram_txt,dim=0))

        return torch.cat(batch_vid_a,dim=0),torch.cat(batch_vid_m,dim=0), torch.cat(batch_smq,dim=0),torch.stack(batch_fw,dim=0)
             

class HME_D(nn.Module):

    def __init__(self, hidden_size, answer_vocab_size):

        """Set the hyper-parameters and build the layers."""
        super(HME_D, self).__init__()
                
        self.linear_decoder_att_a = nn.Linear(hidden_size * 2, hidden_size) 
        self.linear_decoder_att_m = nn.Linear(hidden_size * 2, hidden_size) 
        self.linear_decoder_mem = nn.Linear(hidden_size * 2, hidden_size) 
        

        self.linear_decoder_count_2 = nn.Linear(hidden_size * 4, answer_vocab_size)
   


    def forward(self, vid_att_a,vid_att_m,smq,feature_vqac):   
        ######################### 
        # decode the final output
        ######################### 

        final_embed_a = torch.tanh( self.linear_decoder_att_a(vid_att_a) )
        final_embed_m = torch.tanh( self.linear_decoder_att_m(vid_att_m) )
        final_embed_2 = torch.tanh( self.linear_decoder_mem(smq) )

        final_embed = torch.cat([final_embed_a,final_embed_m,final_embed_2,feature_vqac],dim=1)

        outputs = self.linear_decoder_count_2(final_embed)
        _,predictions = torch.max(outputs,1)
        

        
        return outputs, predictions
    



