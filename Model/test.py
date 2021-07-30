import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from dataset import Dataset , Datasettest
import pdb
from pandas import DataFrame
import json
import pandas as pd

from model import VideoQaNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getInput(vid,next,mv,questions, answers, question_lengths, encode_answer=True):

    bsize = len(vid)
    vid = np.array(vid).astype(np.float32)
    next = np.array(next).astype(np.float32)
    mv = np.array(mv).astype(np.float32)
    questions = np.array(questions).astype(np.int64)
    
    if encode_answer:
        answers = np.array(answers).astype(np.int64)
        answers = torch.from_numpy(answers).to(device,non_blocking=True)

    
    else:
        answers=[]

    vid = torch.from_numpy(vid).to(device)
    next = torch.from_numpy(next).to(device) 
    mv = torch.from_numpy(mv).to(device)         
    question_words = torch.from_numpy(questions).to(device)

    
    return vid,next,mv,question_words,question_lengths,answers

def main():
    """Main script."""
    torch.manual_seed(1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='train/test')
    parser.add_argument('--save_path', type=str, default='./saved_models/',
                        help='path for saving trained models')
    parser.add_argument('--test', type=int, default=0, help='0 | 1')
    args = parser.parse_args()
    
       
    args.word_dim = 300
    args.vocab_num = 8000
    args.pretrained_embedding = '/mnt2/VQA/VideoQA/data/msrvtt_qa/word_embedding.npy'
    args.video_feature_dim = 4096
    args.video_feature_num = 20
    args.answer_num = 1000
    args.memory_dim = 256
    args.batch_size = 1
    args.reg_coeff = 1e-5
    args.learning_rate = 0.0001
    args.log = './logs'
    args.hidden_size = 512
    #args.image_feature_net = 'concat'
    #args.layer = 'fc'
    dataset = Dataset(args.batch_size, './data/',20)
    datasettest = Datasettest(1, './data/',20)
    args.save_model_path = args.save_path + 'model_'
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)



    #############################
    # get video feature dimension
    #############################
    feat_channel = args.video_feature_dim
    feat_dim = 1
    text_embed_size = args.word_dim
    vocab_size = args.answer_num
    voc_len = args.vocab_num
    num_layers = 2
    max_sequence_length = args.video_feature_num
    word_matrix = np.load(args.pretrained_embedding)
    answerset = pd.read_csv('./data/answer_set.txt', header=None)[0]
    qa_type='memory'
    vi_type='memory'
    batch_size = args.batch_size
    model = VideoQaNet(512,20,512,  text_embed_size, 8000, word_matrix, 20,1000,  qa_type, vi_type,2,3)
    model = model.cuda()
    resultss=DataFrame(columns=['id','correct','answer0','answer1',
                'answer2','answer3','answer4','answer5','answer6','answer7','answer8','answer9'])
    

    #resultss = resultss.append(save_data,ignore_index=True)
    
    #if args.test == 1:
    model.load_state_dict(torch.load(os.path.join(args.save_model_path, 'model-2200-_0.368.pkl')))

    # loss function
    criterion = nn.CrossEntropyLoss(size_average=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    #pdb.set_trace()
    
    iter = 0
    
        
    model.eval()
    with torch.no_grad():
        idx = 0
        correct=0
        alpha1_a =[]
        alpha2_a = []
        alpha_a = []
        while datasettest.has_test_example:
            appear, appear_next,mv, questions, answers, question_lengths,ex_id = datasettest.get_test_example()
            vid,next,mv,question_words,question_lengths,answers_ = getInput(appear, appear_next,mv, [questions], answers, question_lengths,encode_answer=False)
                        
            outputs, predictions,att1,att2,alpha1, alpha2, alpha3 = model(vid, next,mv, question_words, question_lengths)
            prediction = predictions
            #pdb.set_trace()
            idx += 1
            if answerset[prediction[0].item()] == answers:
                correct += 1
            alpha1_a.append(alpha1.clone().cpu().numpy())
            alpha2_a.append(alpha2.clone().cpu().numpy())
            alpha_a.append(alpha3.clone().cpu().numpy())
            save_data = {'id':ex_id,'correct':answers,'answer0':answerset[prediction[0].item()],
                          'answer1':answerset[prediction[1].item()],
                          'answer2':answerset[prediction[2].item()],'answer3':answerset[prediction[3].item()],
                          'answer4':answerset[prediction[4].item()],
                          'answer5':answerset[prediction[5].item()],'answer6':answerset[prediction[6].item()],
                          'answer7':answerset[prediction[7].item()],'answer8':answerset[prediction[8].item()],
                          'answer9':answerset[prediction[9].item()]}
            print(save_data)
            resultss = resultss.append(save_data,ignore_index=True)
        test_acc = 1.0*correct / dataset.test_example_total
        print (correct, dataset.test_example_total)
        print('Test[%d] video %d, acc %.3f' % (iter,example_id, test_acc))
        datasettest.reset_test()
        savename = 'fix_50_test.json'
        pdb.set_trace()    
        resultss.to_json(savename,'records')        

            


if __name__ == '__main__':
    main()
