# CogDL at MAG240M
This is an implementation of CogDL for mag240m.

## Getting Started
### Requirements
* Python==3.7
* PyTorch==1.9.0+cu111
* dgl-cuda111==0.7.0
* ogb==1.3.4
* sklearn==1.0.2
<!-- * tsc-auto==0.28 -->


## Running this code
### Preprocess graph and generate features
```shell
bash run_preprocess.sh
``` 
After this step, you will get ```.npy``` feature files in ```MAG_FEAT_PATH```. 
### Get embeddings of other methods:
We extract the metapath2vec embeddings of arXiv papers, which are provided by [R-UNIMP](https://github.com/PaddlePaddle/PGL/tree/main/examples/kddcup2021/MAG240M/r_unimp). We also try to reproduce the code of [Deepmind](https://github.com/deepmind/deepmind-research/tree/master/ogb_lsc/mag) and gain the arXiv paper embedding. 
The arXiv paper embeddings of the above two methods can be obtained from the following link (password 3oot):

https://pan.baidu.com/s/1EVlhs2jClCJHwbH-ht2vqw 

Please download the ```x_m2v_64.npy``` and ```x_jax_153.npy```. Then, put them into ```MAG_FEAT_PATH```.

The pre-processing and generating features are time-consuming, and we also provide all features as downloadable options so you can choose the ones you need.

### Train our model
We selected different features for different models, the corresponding relationships are listed in ```features/config_feats.py```.
```shell
bash run_folds.sh
``` 

### Run ensemble
```shell
bash run_ensemble.sh
``` 
This step contains 15 models (14 trained models + Deepmind embedding), for more details please refer to our technical report.