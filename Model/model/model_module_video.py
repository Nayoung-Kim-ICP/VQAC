import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)

class VideoEncoderLSTM(nn.Module):
    def __init__(self,input_size,input_number,hidden_size,feature_spatial_size,feature_channel):
        super(VideoEncoderLSTM, self).__init__()

        self.input_number = input_number
        self.hidden_size = hidden_size

        self.FrameMotion = torch.nn.Sequential()
        self.FrameMotion.add_module('res1', conv3x3(n_feats*2,n_feats,requires_grad=True))
        self.FrameMotion.add_module('relu1', nn.ReLU(inplace=True))
        self.FrameMotion.add_module('res2', conv3x3(n_feats,n_feats,requires_grad=True))
        self.FrameMotion.add_module('relu2', nn.ReLU(inplace=True))

        self.lstm_video_1a = nn.LSTMCell(hidden_size*2, hidden_size)
        self.lstm_video_2a = nn.LSTMCell(hidden_size, hidden_size)
        
        self.lstm_video_1m = nn.LSTMCell(hidden_size*2, hidden_size)
        self.lstm_video_2m = nn.LSTMCell(hidden_size, hidden_size)

        self.FC_1 = nn.Linear(feature_spatial_size*feature_channel,hidden_size*2)
        self.FC_2 = nn.Linear(hidden_size*2,hidden_size)

        self.sigmoid = nn.Sigmoid()

    def init_hiddens(self):
        s_t = torch.zeros(1, self.hidden_size).cuda()
        s_t2 = torch.zeros(1, self.hidden_size).cuda()
        c_t = torch.zeros(1, self.hidden_size).cuda()
        c_t2 = torch.zeros(1, self.hidden_size).cuda()
        return s_t,s_t2,c_t,c_t2

    def ALPHA(self,resi):

        similarity = att*motion
        wegigh = self.sigmoid(similarity)
        return wegigh*motion


    def forward(self, f_t,f_t_1, mv,att1,att2):
        apperance_batch = []
        motion_batch = []
        batch_size = apperances.shape[0]
        f_s = f_t
        f_m = self.FrameMotion(f_t,f_t_1)

        for j in range(batch_size):
            motions,alpha = self.FrameMotion(apperances[j,:], apperances_next[j,:], mv[j,0,:])
            apperance = apperances[j:j+1,:]
            #motion = motions[j:j+1,:]

            s1_t1_m,s1_t2_m,c1_t1_m,c1_t2_m = self.init_hiddens()
            s1_t1_a,s1_t2_a,c1_t1_a,c1_t2_a = self.init_hiddens()   
            all_m = []
            all_a = []
            for i in range(self.input_number):
                input_motion = torch.cat([self.ATT(att1[j:j+1],motions[:,i]),motions[:,i]],dim=1)
                tapp=apperance[:,i]
                appear=tapp.view(tapp.size(0),tapp.size(1)*tapp.size(2)*tapp.size(3))
                appear=self.FC_2(F.relu(self.FC_1(appear)))
                input_apperance = torch.cat([self.ATT(att2[j:j+1],appear),appear],dim=1)
                s1_t1_m, c1_t1_m = self.lstm_video_1m(input_motion, (s1_t1_m, c1_t1_m))
                s1_t2_m, c1_t2_m = self.lstm_video_2m(s1_t1_m, (s1_t1_m, c1_t2_m))
            
                s1_t1_a, c1_t1_a = self.lstm_video_1a(input_apperance, (s1_t1_a, c1_t1_a))
                s1_t2_a, c1_t2_a = self.lstm_video_2a(s1_t1_a, (s1_t2_a, c1_t2_a))
                all_m.append(torch.cat((s1_t1_m,s1_t2_m),dim=1))
                all_a.append(torch.cat((s1_t1_a,s1_t2_a),dim=1))
            # here s1_t1, s1_t2 is the last hidden
            #s1_t_a = torch.cat( (s1_t1_a,s1_t2_a), dim=1) 
            #s1_t_m = torch.cat( (s1_t1_m,s1_t2_m), dim=1) 
            apperance_batch.append(torch.stack(all_a,dim=1))
            motion_batch.append(torch.stack(all_m,dim=1))
        return torch.cat(apperance_batch, dim=0),torch.cat(motion_batch, dim=0),alpha


class VideoEncoderMemory(nn.Module):
    def __init__(self,input_size,input_number,hidden_size,batch_size,max_len=20):
        super(VideoEncoderMemory, self).__init__()
        self.batch_size = batch_size
        self.input_number = input_number
        self.motion_module = True
        self.FrameMotion = FrameMotion(batch_size,hidden_size)
        self.lstm_video_1a = nn.LSTMCell(hidden_size*2, hidden_size)
        self.lstm_video_2a = nn.LSTMCell(hidden_size, hidden_size)
        self.MV = nn.Linear(49,hidden_size)
        self.MV2 = nn.Linear(hidden_size*2,hidden_size)
        self.lstm_video_1m = nn.LSTMCell(hidden_size*2, hidden_size)
        self.lstm_video_2m = nn.LSTMCell(hidden_size, hidden_size)
        self.apperances_module = True
        self.hidden_size=hidden_size
        self.mrm_vid = MemoryRamTwoStreamModule(hidden_size, hidden_size, max_len)
    def init_weights(self, word_matrix):
        """Initialize weights."""

        if word_matrix is None:
            self.embed.weight.data.uniform_(-0.1, 0.1)
        else:
            # init embed from glove
            self.embed.weight.data.copy_(torch.from_numpy(word_matrix))
        self.mrm_vid.init_weights()


    def init_hiddens(self):
        s_t = torch.zeros(1, self.hidden_size).cuda()
        s_t2 = torch.zeros(1, self.hidden_size).cuda()
        c_t = torch.zeros(1, self.hidden_size).cuda()
        c_t2 = torch.zeros(1, self.hidden_size).cuda()
        return s_t,s_t2,c_t,c_t2

    def forward(self, apperances,motions):

        memory_ram_vid_batch=[]
        for j in range(self.batch_size):

            motion = motions[j:j+1,:]
            apperance = apperances[j:j+1,:]

            s1_t1_m,s1_t2_m,c1_t1_m,c1_t2_m = self.init_hiddens()
            s1_t1_a,s1_t2_a,c1_t1_a,c1_t2_a = self.init_hiddens()   
            hidden_array_2m=[]
            hidden_array_2a=[]
            for i in range(self.input_number):
                input_motion = motion[:,i]
                s1_t1_m, c1_t1_m = self.lstm_video_1m(input_motion, (s1_t1_m, c1_t1_m))
                s1_t2_m, c1_t2_m = self.lstm_video_2m(s1_t1_m, (s1_t2_m, c1_t2_m))
                sV_t2m_vec = s1_t2_m.view(s1_t2_m.size(0),1,s1_t2_m.size(1))
                hidden_array_2m.append(sV_t2m_vec)

                input_apperance = apperance[:,i]
                s1_t1_a, c1_t1_a = self.lstm_video_1a(input_apperance, (s1_t1_a, c1_t1_a))
                s1_t2_a, c1_t2_a = self.lstm_video_2a(s1_t1_a, (s1_t2_a, c1_t2_a))
                sV_t2a_vec = s1_t2_a.view(s1_t2_a.size(0),1,s1_t2_a.size(1))
                hidden_array_2a.append(sV_t2a_vec)
                

            sV_l2a = torch.cat(hidden_array_2a, dim=1)
            sV_l2m = torch.cat(hidden_array_2m, dim=1)
            memory_ram_vid = self.mrm_vid(sV_l2a[0,:,:], sV_l2m[0,:,:], self.input_number)
            memory_ram_vid_batch.append(memory_ram_vid)
            #pdb.set_trace()
        return torch.stack(memory_ram_vid_batch,dim=0)




 
class MultiModalNet(nn.Module):
    def __init__(self,hidden_size, version,loop):
        super(MultiModalNet, self).__init__()
        self.version = version
        self.mm_att = MultiModalAttentionModule(hidden_size)
        self.hidden_size = hidden_size
        self.loop = loop
        self.softmax = torch.nn.Softmax(dim=1)
        if self.version ==1:
            self.lstm_mm_1 = nn.LSTMCell(hidden_size, hidden_size)
            self.lstm_mm_2 = nn.LSTMCell(hidden_size, hidden_size)
            self.Linear_decoder_mem = nn.Linear(hidden_size * 2, hidden_size) 
            self.hidden_encoder_1 = nn.Linear(hidden_size * 2, hidden_size) 
            self.hidden_encoder_2 = nn.Linear(hidden_size * 2, hidden_size) 
        
        elif self.version ==2:
            self.gru_mm = nn.GRUCell(hidden_size, hidden_size)
            self.Linear_decoder_mem = nn.Linear(hidden_size, hidden_size) 
        else:
            self.text_enc =  nn.Linear(input_size , hidden_size)
            self.qa_enc =  nn.Linear(hidden_size , hidden_size)   
            self.ans_lstm = nn.LSTMCell(hidden_size, hidden_size)
            self.ch_weight = nn.LSTMCell(hidden_size, 2)

    def init_hiddens(self):
        s_t = torch.zeros(1, self.hidden_size).cuda()
        s_t2 = torch.zeros(1, self.hidden_size).cuda()
        c_t = torch.zeros(1, self.hidden_size).cuda()
        c_t2 = torch.zeros(1, self.hidden_size).cuda()
        return s_t,s_t2,c_t,c_t2

    def attend(self,target, sources):
        w = self.softmax(torch.sum(target.view(target.size(0),target.size(1),1)*sources,dim=2))
        a = torch.sum(w.view(w.size(0),w.size(1),1)*sources,dim=1)
        return w, a

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
    
    
    def mm_module_v2(self,memory_ram_vid,memory_ram_txt,loop=5):
        
        h_t = torch.zeros(1, self.hidden_size).cuda()
                
        for _ in range(loop):
            mm_o = self.mm_att(h_t,memory_ram_vid,memory_ram_txt)
            h_t = self.gru_mm(mm_o, h_t)
        
        return h_t

    def mm_module_v3(self,embed,enc_text,appear,motion,enc_cell):
        
        Fusion = torch.zeros(self.batch_size, self.hidden_size).cuda()
        sm_q1,_,cm_q1,_ = self.init_hiddens()
    
        for i in range(len(embed[0])):
            word_embed = embed[:,i]
            word_vec = torch.nn.tanh(self.text_enc(word_embed))
            qa_vec = torch.nn.tanh(self.qa_enc(enc_text[:,i]))
            #first
            app_weight, app_att = self.attend(word_vec,appear)
            mot_weight, mot_att = self.attend(word_vec,motion)
            channel_weight = self.softmax(self.ch_weight(word_embed))
            cw_app = channel_weight[:,0:1]
            cw_mot = channel_weight[:,1:2]
            cc_video = cw_app*app_att+mot_weight*mot_att
            pv_video = fusion
            I_input = pv_video + cc_video+ qa_vec
            sm_q1,cm_q1 = self.answer_lstm(I_input,(sm_q1,cm_q1))
            #sec
            app_weight_2, _ = self.attend(sm_q1,appear)
            mot_weight_2, _ = self.attend(sm_q1,motion)

            fin_app = (app_weight_2+app_weight)
            fin_mot = (mot_weight_2+mot_weight)

            app = appear*fin_app
            mot = motion*fin_mot
            ch_w = self.ch_weight(enc_text[:,i])
            fusion = fin_app*ch_w[:,0:1]+fin_mot*ch_w[:,1:2]

        q_info = torch.nn.tanh(enc_cell)
        a_info = torch.nn.tanh(cm_q1)
        v_info =  torch.nn.tanh(fusion)
        return q_info*a_info*v_info


    def forward(self, embed, memory_vid, memory_text,appear,motion,enc_cell):
        if self.version==1:
            output = self.mm_module_v1(self.hidden_state, memory_vid, memory_text)
        elif self.version==2:
            output = self.mm_module_v2(memory_vid, memory_text)
        else:
            output = self.mm_module_v3(embed,hidden_state,appear,motion,enc_cell,self.loop)       
        return output

class motionFeature(nn.Module):
    def __init__(self,input_size):
        super(motionFeature, self).__init__()
        self.conv_m = torch.nn.Sequential() 
        self.conv_m.add_module('conv1', nn.Conv2d(in_channels=input_size*2, out_channels=input_size, kernel_size=1, stride=1, padding=0)) 
        self.conv_m.add_module('relu1', nn.ReLU(inplace=True))

    def forward(self, f_s,f_s_1):
        cat_feature = torch.cat([f_s,f_s_1],dim=2)
        in_feature = cat_feature.view(cat_feature.size(0)*cat_feature.size(1),cat_feature.size(2),cat_feature.size(3),cat_feature.size(4))
        out_feature = self.conv_m(in_feature)
        f_m = out_feature.view(f_s.size(0),f_s.size(1),f_s.size(2),f_s.size(3),f_s.size(4))
        return f_m


class QuestionguidedAtt(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(QuestionguidedAtt, self).__init__()

        self.W3 = nn.Conv2d(in_channels=input_size, out_channels=hidden_size, kernel_size=1, stride=1, padding=0)
        self.W4 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, stride=1, padding=0)
        self.W2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, stride=1, padding=0)
        self.tanh = torch.nn.Tanh()
        
    def forward(self, f_s,f_m,Eq):
        batch_as = [] 
        batch_am = []

        len_v = f_s.size(1)
        w_weight = self.W2(Eq)
        
        for i in range(len_v):
            inz=torch.add(self.W3(f_s[:,i,:,:,:]),w_weight)
            z_s = self.W4(self.tanh(inz))
            inz_= torch.add(self.W3(f_m[:,i,:,:,:]),w_weight)
            z_m = self.W4(self.tanh(inz_))
            t_z_s = z_s.view(z_s.size(0),z_s.size(1),z_s.size(2)*z_s.size(3))
            t_A_s = torch.softmax(t_z_s, dim=2)
            A_s = t_A_s.view(z_s.size(0),z_s.size(1),z_s.size(2),z_s.size(3))
            t_z_m = z_m.view(z_m.size(0),z_m.size(1),z_m.size(2)*z_m.size(3))
            t_A_m = torch.softmax(t_z_m, dim=2)
            A_m = t_A_m.view(z_m.size(0),z_m.size(1),z_m.size(2),z_m.size(3))
            batch_as.append(A_s)
            batch_am.append(A_m)

        A_s = torch.stack(batch_as,dim=1)
        A_m = torch.stack(batch_am,dim=1) 

        return A_s*f_s, A_m*f_m



class ResidueWeightedvectorG(nn.Module):
    def __init__(self,input_size,input_number,hidden_size,answer_size,video_mode):
        super(ResidueWeightedvectorG, self).__init__()
        
        self.RWvec = ResidueWeightedvector(49,input_size,hidden_size)
        self.Decoder = Decoder(hidden_size,answer_size)
        self.W7 = nn.Linear(hidden_size,hidden_size)
        self.W8 = nn.Linear(hidden_size,hidden_size)
        self.vid_fusion_mode = video_mode

        if video_mode == 'lstm':
            self.lstm_video_1m = nn.LSTMCell(hidden_size, hidden_size)
            self.lstm_video_2m = nn.LSTMCell(hidden_size, hidden_size)
            self.hidden_size = hidden_size

    def init_hiddens(self):
        s_t = torch.zeros(1, self.hidden_size).cuda()
        s_t2 = torch.zeros(1, self.hidden_size).cuda()
        c_t = torch.zeros(1, self.hidden_size).cuda()
        c_t2 = torch.zeros(1, self.hidden_size).cuda()
        return s_t,s_t2,c_t,c_t2

    def forward(self, g_s,g_m,residue,e_t):
        output_batch = []
        max_batch = []
        batch_size = g_s.shape[0]

        for j in range(batch_size):

            motions = self.RWvec(g_s[j,:], g_m[j,:], residue[j,:])
            ### video feature -> temporal fusion ###
            if self.vid_fusion_mode == 'sum':
                l_alpha = torch.sum(motions,dim=0, keepdim=True)

            elif self.vid_fusion_mode == 'lstm':
                s1_t1,s1_t2,c1_t1,c1_t2 = self.init_hiddens()
                for vidt in range(motions.shape[0]):
                    sV_t1m, cV_t1m = self.lstm_video_1m(motions[vidt:vidt+1,:], (s1_t1, c1_t1))
                    sV_t2m, cV_t2m = self.lstm_video_2m(sV_t1m, (s1_t2, c1_t2))
                l_alpha = sV_t2m

            dec_l_alpha = self.W7(l_alpha)
            dec_t = self.W8(e_t[j,])
        
            Fusion= dec_l_alpha+dec_t
            output_batch.append(Fusion)

        decoder_feature = torch.cat(output_batch,axis=0)

        output,max_id=self.Decoder(decoder_feature)
        return decoder_feature,output,max_id



class ResidueWeightedvector(nn.Module):
    def __init__(self, resi_in,input_size,hidden_size):
        super(ResidueWeightedvector, self).__init__()

        self.alpha_layer = nn.Linear(resi_in,1)
        self.th = 1
        self.W9 = torch.nn.Conv2d(in_channels=input_size, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.FC_m1 = nn.Linear(28*28*4,hidden_size*2)
        self.FC_m2 = nn.Linear(hidden_size*2,hidden_size)
        

    def alpha(self,x):
        output = torch.sigmoid(self.alpha_layer(x))
        return output


    def forward(self, g_s, g_m, Residue): 
        alpha_t = self.alpha(Residue)
        
        alpha_t=alpha_t.view(alpha_t.size(0),1,1,1)

        input_fc = (1-alpha_t)*g_s+(alpha_t)*g_m
        input_fc = F.relu(self.W9(input_fc))
        input_fc = input_fc.view(input_fc.size(0),input_fc.size(1)*input_fc.size(2)*input_fc.size(3))
        l_v = self.FC_m2(F.relu(self.FC_m1(input_fc)))
        return l_v


class Decoder(nn.Module):
    def __init__(self,hidden_size,answer_size):
        super(Decoder, self).__init__()
        self.FC1 = nn.Linear(hidden_size,hidden_size)
        self.FC2 = nn.Linear(hidden_size,answer_size) 

    def forward(self, multimodal_input):
     
        outputs = F.relu(self.FC1(multimodal_input))
        outputs = self.FC2(outputs)
        _,mx_idx = torch.max(outputs,1)
        
        return outputs, mx_idx

