# DONet
# DONet: Dual-Octave Network for Fast MR Image Reconstruction (IEEE Transactions on Neural Networks and Learning Systems)

## Dependencies
* Python 3.7
* Tensorflow 1.14
* numpy
* h5py
* skimage
* matplotlib
* tqdm

Install dependencies as follows:
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
bash Anaconda3-2020.07-Linux-x86_64.sh.sh
source ~/.bashrc
conda install python=3.7
conda install tensorflow-gpu==1.14
conda install numpy
conda install scikit-learn
conda install scikit-image
conda install tqdm
conda install opencv
```

## Project Structure
DONet
  ├──README.md
  ├──code
    ├──data_preparation
    └──model
  ├──experiment
  └──results

## Dataset and Prepartion
All data that we used for our experiments are released at GLOBUS(https://app.globus.org/file-manager?origin_id=15c7de28-a76b-11e9-821c-02b7a92d8e58&origin_path=%2F).
Before training, we recommend you to process data into ```.tfrecords``` to accelerate the progress.  File ```./data_preparation/data2tfrecords.py``` specifies the route of data processing.

## How to train and test on DONet
Unpack the dataset file to the folder you defined. Then, change the ```data_dst``` argument in ```./option.py``` to the place where datasets are located.

Enter in the folder ```/DONet/code```

**Train**
```bash
CUDA_VISIBLE_DEVICES=0 python run.py --n_GPU 1 --name Dual-Oct_dense_B10_lrb3_a0.125_cpd320_1Un3X --n_blocks 10 --n_feats 64 --lr 1e-3 --alpha 0.125 --data_dst coronal_pd_320 --epoch 50 --mask_name 1Un3_320
```

**Test**
```bash
CUDA_VISIBLE_DEVICES=0 python tester.py --n_GPU 1 --rsname Dual-Oct_dense_B10_lrb3_a0.125_cpd320_1Un3X --n_blocks 10 --n_feats 64 --alpha 0.125 --data_dst coronal_pd_320 --mask_name 1Un3_320 --test_only --save_gt --save_results
```

Change other arguments in option.py or in the shell that you can train your own model.

If one GPU will be out of memory, you can change the ```--n_blocks``` and  ```--n_feats``` to compress the model, empirically we set ```--n_resblocks 10``` and ```--n_feats 64```. Moreover, you can set ```CUDA_VISIBLE_DEVICES=0, 1```, ```--n_GPU 2```and try using two or more GPUs for this training.

Citation

If you find DONet useful for your research, please consider citing the following papers:

```
@inproceedings{feng2021DONet,
  title={DONet: Dual-Octave Network for Fast MR Image Reconstruction},
  author={Feng, Chun-Mei and Yang, Zhanyuan and Fu, Huazhu and Xu, Yong and Yang, Jian and Shao, Ling},
  booktitle={IEEE Transactions on Neural Networks and Learning Systems},
  year={2021}
}
```
