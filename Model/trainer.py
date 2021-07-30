from utils import *
from os import listdir
from os.path import join
from pandas import DataFrame
import os
import numpy as np
import torch 
import torch.optim as optim
import torchvision.utils as utils



def accuracy( logits, targets):
    correct = torch.sum(logits.eq(targets)).float()
    return correct * 100.0 / targets.size(0)

class Trainer():
    def __init__(self, args, logger, dataloader, model,answerset):
        
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.model = model
        self.answerset = answerset
        
        self.device = torch.device('cpu') if args.cpu else torch.device('cuda')
        self.load(model_path=self.args.model_path)
        

        self.params = [
            {"params": filter(lambda p: p.requires_grad, self.model.parameters() if 
             args.num_gpu==1 else self.model.module.parameters()), 
             "lr": args.lr_rate
            }
        ]   

    # loss function
        self.loss = nn.CrossEntropyLoss(size_average=True).to(self.device)
        self.optimizer = optim.Adam(self.params, betas=(args.beta1, args.beta2), eps=args.eps)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.decay, gamma=self.args.gamma)

    
    def load(self, model_path=None):
        if (model_path):
            self.logger.info('load_model_path: ' + model_path)
            if listdir(model_path)!=[]:
                path=join(model_path,listdir(model_path)[0])
                model_state_dict_save = {k:v for k,v in torch.load(path, map_location=self.device).items()}
                model_state_dict = self.model.state_dict()
                model_state_dict.update(model_state_dict_save)
                self.model.load_state_dict(model_state_dict)
                print(path)
            else:
                self.logger.info('There are no weights of a mix model to load in this path')


        if ((not self.args.cpu) and (self.args.num_gpu > 1)):

            self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))

    def prepare(self, sample_batched):
        for key in sample_batched.keys():
            sample_batched[key] = sample_batched[key].to(self.device)
        return sample_batched
                

    def train(self, current_epoch=0, is_init=False):
        self.model.train()
        is_print=0
        if (not is_init):
            self.scheduler.step()
        self.logger.info('Current epoch learning rate: %e' %(self.optimizer.param_groups[0]['lr']))

        for i_batch, sample_batched in enumerate(self.dataloader['train']):

            self.optimizer.zero_grad()
            
            sample_batched = self.prepare(sample_batched)
            #pdb.set_trace()
            f_t = sample_batched['f_t']
            f_t_1 = sample_batched['f_t_1']
            r_t = sample_batched['r_t']
            questions = sample_batched['questions']
            answers = sample_batched['answers']
            question_lengths = sample_batched['question_lengths']
            
            
            outputs, predictions = self.model(f_t, f_t_1,r_t, questions, question_lengths)
            
            targets = answers

            loss = self.loss(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            acc = accuracy(predictions, targets)
            is_print +=1 
            
            if ( is_print % 10 == 0):
                self.logger.info('saving the model...')
                val_acc = self.test()
                tmp = self.model.state_dict()
                model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp }            
                model_name = self.args.model_path.strip('/')+'/model_'+str(current_epoch).zfill(5)+'_'+str(round(val_acc,3))+'_.pt'
                torch.save(model_state_dict, model_name)

            """



            if (is_print%100==0):
                self.logger.info( ('init ' if is_init else '') + 'epoch: ' + str(current_epoch) + 
                    '\t batch: ' + str(i_batch+1) )

                self.logger.info( 'loss: %.6f, acc: %.4f' %(loss.item(),acc.item()) )


  
        if ( current_epoch % 1 == 0):
            self.logger.info('saving the model...')
            val_acc = self.test()
            tmp = self.model.state_dict()
            model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp }            
            model_name = self.args.model_path.strip('/')+'/model_'+str(current_epoch).zfill(5)+'_'+str(round(val_acc,3))+'_.pt'
            torch.save(model_state_dict, model_name)
            d"""  
            
    def val(self,current_epoch):
        self.logger.info('Epoch ' + str(current_epoch) + ' validation process...')
        self.model.eval()
        with torch.no_grad():
            idx = 0
            correct = 0 
            for i_batch, sample_batched in enumerate(self.dataloader['val']):
                sample_batched = self.prepare(sample_batched)

                f_t = sample_batched['f_t']
                f_t_1 = sample_batched['f_t_1']
                r_t = sample_batched['r_t']
                questions = sample_batched['question']
                answers = sample_batched['answer']
                question_lengths = sample_batched['question_length']
                test_example_total = sample_batched['test_example_total']
                output, prediction = self.model(f_t, f_t_1,r_t, questions, question_lengths)
                prediction = prediction.item()
                idx += 1
                if self.answerset[prediction] == answers:
                    correct += 1

            test_acc = 1.0*correct / test_example_total
            
            print('Val[%d] : acc %.3f' % (current_epoch, test_acc))
            
        self.model.train()

    def test(self):

        results = DataFrame(columns=['id','gt','answer'])
        self.logger.info( ' test process...')
        self.model.eval()
        with torch.no_grad():
            idx = 0
            correct = 0 

            for i_batch, sample_batched in enumerate(self.dataloader['test']):

                sample_batcheds = self.prepare(sample_batched[0])

                question_lengths = sample_batched[1]
                v_id = sample_batched[2]
                answer = sample_batched[3]
                quesetion = sample_batched[4]
                test_example_total = sample_batched[5]


                f_t = sample_batcheds['f_t']
                f_t_1 = sample_batcheds['f_t_1']
                r_t = sample_batcheds['r_t']
                questions = sample_batcheds['question']
                
                output, prediction = self.model(f_t, f_t_1,r_t, questions, question_lengths)
                prediction = prediction.item()
                pred_answer = self.answerset[prediction]
                save_data = {'id':v_id[0].item(),'gt':answer[0],'answer':pred_answer}

                results = results.append(save_data,ignore_index=True)
                idx += 1
                if self.args.is_print_mode:
                    print('[Vid',v_id[0].item(),'] Q: ',quesetion[0],'Ans: ', self.answerset[prediction],'GT: ',answer[0])
                if pred_answer == answer[0]:
                    correct += 1

            test_acc = 1.0*correct / idx
            save_name = 'result%.3f.json'%(test_acc)
            results.to_json(save_name,'records')
            print('Test : acc %.3f' % ( test_acc))

        self.model.train()
        return test_acc

            

        dataset.reset_test()
        print('correct',correct / idx )
        print('end')
        

