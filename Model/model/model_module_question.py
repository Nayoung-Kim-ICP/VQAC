import torch
import torch.nn as nn
import torch.nn.functional as F



import pdb
import numpy as np






class QuestionEncoderLSTM(nn.Module):
    def __init__(self, hidden_size,text_embed_size,embed):
        """Set the hyper-parameters and build the layers."""
        super(QuestionEncoderLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.embed = embed

        self.E_W = nn.Linear(self.hidden_size,28*28)
        self.lstm_text_1 = nn.LSTMCell(text_embed_size, self.hidden_size)
        self.lstm_text_2 = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.softmax = torch.nn.Softmax(dim=1)

    def init_hiddens(self):
        s_t = torch.zeros(1, self.hidden_size).cuda()
        s_t2 = torch.zeros(1, self.hidden_size).cuda()
        c_t = torch.zeros(1, self.hidden_size).cuda()
        c_t2 = torch.zeros(1, self.hidden_size).cuda()
        return s_t,s_t2,c_t,c_t2
    

    def forward(self, questions,question_lengths):

        features_questions = self.embed(questions)
        batch_size = len(questions)

        # question encoding
        batch_s1_t1 = []
        batch_s1_t2 = []
        batch_c1_t1 = []
        batch_c1_t2 = []
        for j in range(batch_size):            
            nQuestionWords = question_lengths[j]
            s1_t1,s1_t2,c1_t1,c1_t2 = self.init_hiddens()
            
            for i in range(nQuestionWords):
                input_question = features_questions[j,i:i+1,:]
                
                s1_t1, c1_t1 = self.lstm_text_1(input_question, (s1_t1, c1_t1))
                s1_t2, c1_t2 = self.lstm_text_2(s1_t1, (s1_t2, c1_t2))
                

            batch_s1_t1.append(s1_t1)
            batch_s1_t2.append(s1_t2)
            batch_c1_t1.append(c1_t1)
            batch_c1_t2.append(c1_t2)
        
        batch_s1_t1 = torch.stack(batch_s1_t1, dim=0)
        batch_s1_t2 = torch.stack(batch_s1_t2, dim=0)
        batch_c1_t1 = torch.stack(batch_c1_t1, dim=0)
        batch_c1_t2 = torch.stack(batch_c1_t2, dim=0)
        E_q = torch.tile(batch_s1_t2,(1,28*28,1))#self.E_W(f_w_sum)
        E_q = E_q.view(-1,28,28,E_q.size(2))
        E_q = torch.transpose(E_q,1,3)

        return E_q, [batch_s1_t1,batch_s1_t2,batch_c1_t1,batch_c1_t2]

