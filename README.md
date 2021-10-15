### Video Question Answering Using Language-Guided Deep Compressed-Domain Video Feature
### (VQAC)

This is the PyTorch Implementation of 
* Nayoung Kim, Seong-Jong Ha, and Je-Won Kang. *Video Question Answering Using Language-Guided Deep Compressed-Domain Video Feature*. In *ICCV*, 2021. (to appear)

### Download preprocessing data
In this experiment, we use [MSVD-QA dataset](https://github.com/xudejing/VideoQA).
Please refer to their website for the detailed statistics of this dataset.

We already upload compressed-domain video features.
You don't need to download original videos.
~~~
cd Model
~~~

### Preprocessing
If you want to generate features, follow the below step. (Will be)
1. Video encoding
 To extract motion vector and residue by HM 16.04, you need to follow this process:
 - resize the video resolution: 224x224

2. Feature warping




