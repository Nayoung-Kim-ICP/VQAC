import argparse

parser = argparse.ArgumentParser(description='VQAC')

### model setting
parser.add_argument('--word_dim', type=int, default=300,
                    help='Word vector dimension size')
parser.add_argument('--vocab_num', type=int, default=4000,
                    help='Word dimension size')
parser.add_argument('--video_feature_dim', type=int, default=4096,
                    help='Video feature dimension size')
parser.add_argument('--video_feature_num', type=int, default=20,
                    help='Limitation of GOP size')
parser.add_argument('--answer_num', type=int, default=1000,
                    help='Size of answer voca')
parser.add_argument('--memory_dim', type=int, default=256,
                    help='Hidden dimension of memory feature size')
parser.add_argument('--text_embed_size', type=int, default=300,
                    help='Hidden dimension of memory feature size')
parser.add_argument('--hidden_size', type=int, default=512,
                    help='Hidden dimension of memory feature size')
parser.add_argument('--vocab_size', type=int, default=4000,
                    help='Hidden dimension of memory feature size')
parser.add_argument('--input_size', type=int, default=512,
                    help='Hidden dimension of memory feature size')
parser.add_argument('--input_number', type=int, default=512,
                    help='Hidden dimension of memory feature size')
parser.add_argument('--answer_size', type=int, default=1000,
                    help='Answer vocabulary size')
parser.add_argument('--model_mode', type=str, default='HME',
                    help='HME / VQAC')
parser.add_argument('--max_len_qa', type=int, default=20,
                    help='max number of word(question) for training')
parser.add_argument('--max_len_vid', type=int, default=20,
                    help='max number of word(question) for training')
parser.add_argument('--vid_fusion_mode', type=str, default='lstm',
                    help='temporally video fusion mode: sum, lstm')
### path setting

parser.add_argument('--save_path', type=str, default='./saved_model/',
                    help='path for saving trained models')
parser.add_argument('--model_path', type=str, default='./saved_model/model/',
                    help='path for saving trained models')
parser.add_argument('--data_path', type=str, default='./data/',
                    help='path for saving trained models')
parser.add_argument('--qa_path', type=str, default='./data/',
                    help='path for saving trained models')
parser.add_argument('--word_matrix_path', type=str, default='./data/text_data/word_embedding.npy',
                    help='path for word_matrix')
parser.add_argument('--answer_path', type=str, default='./data/text_data/answer_set.txt',
                    help='path for word_matrix')

### mode setting
parser.add_argument('--is_print_mode', type=bool, default=True, 
                    help='print questions and answers when test')
parser.add_argument('--test', type=bool, default=False, 
                    help='Test mode')
parser.add_argument('--reset', type=bool, default=False, 
                    help='Training reset')
parser.add_argument('--is_init', type=bool, default=False, 
                    help='Training reset')
parser.add_argument('--now_epoch', type=int, default=0, 
                    help='current epoch')
parser.add_argument('--num_epoch', type=int, default=100, 
                    help='total epoch')
parser.add_argument('--data_name', type=str, default='msvd', 
                    help='Dataset: MSVD, MSR-VTT')
parser.add_argument('--log_file_name', type=str, default='VQAC.log',
                    help='Log file name')
parser.add_argument('--logger_name', type=str, default='VQAC',
                    help='Logger name')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='The beta1 in Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='The beta2 in Adam optimizer')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='The eps in Adam optimizer')
parser.add_argument('--lr_rate', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--decay', type=float, default=999999,
                    help='Learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='Learning rate decay factor for step decay')
### device setting
parser.add_argument('--cpu', type=bool, default=False, 
                    help='cpu on/off')
parser.add_argument('--num_workers', type=int, default=8, 
                    help='number of worker')

parser.add_argument('--num_gpu', type=int, default=2, 

                    help='number of worker')
args = parser.parse_args()
