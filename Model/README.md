### Video Question Answering Using Language-Guided Deep Conpressed-Domain Video Feature
### (VQAC)

This is the PyTorch Implementation of 
* Nayoung Kim. *Video Question Answering Using Language-Guided Deep Conpressed-Domain Video Feature*. In *ICCV*, 2021. 

### Data 
We provide data and our pre-trained models.
1. Download from [[here]](https://drive.google.com/drive/folders/1WNbZHRVAYIB9BKxO5Y7-matPSAqUvEsk?usp=sharing) and override by follow:

```bash
├── data
│   └── text_data
│       ├── answer_set.txt
│       ├── test_qa_encode.json
│       ├── train_qa_encode.json
│       ├── val_qa_encode.json
│       ├── vocab.txt
│       └── word_embedding.npy
│   └── video_data
│       ├── resi_test.h5
│       ├── resi_train.h5
│       ├── resi_val.h5
│       ├── resnet152_test.h5
│       ├── resnet152_train.h5
│       └── resnet152_val.h5
└── saved_model
    ├── args.txt
    └── model
        └── model_best.pt
``` 

### Train, validate, and test
For testing, execute the following command
~~~~
python main.py --test=True --is_print_mode=True --model_mode=HME
~~~~

For training, execute the following command
~~~~
python main.py --test=Flase --model_mode=VQAC
~~~~
You can select several options in the file :  option.py

### Requirements
Python = 3.6
PyTorch = 1.9 



