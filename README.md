
# On Generalized Degree Fairness in Graph Neural Networks
We provide the implementaion and datasets for our paper "[On Generalized Degree Fairness in Graph Neural Networks](https://arxiv.org/abs/2302.03881)" (DegFairGNN for short), which has been accepted by AAAI 2023.

## 1. Description

The repository is organised as follows:
- data/: contains 3 benchmark datasets: squirrel, chameleon, emnlp. Please extract zip files before running.
- models/: contains our model. 
- layers/: contains component layers for our model.  


## 2. Requirements
To install required packages
- pip3 install -r requirements.txt

## 3. Experiments

To run chameleon:
- python3 main.py --dataset=chameleon --dim=32 --omega=1 --w_f=1e-3

To run squirrel:
- python3 main.py --dataset=squirrel --dim=32 --omega=1 --w_f=1e-4 --epoch=1000 

To run emnlp:
- python3 main.py --dataset=emnlp --dim=16 --omega=0.001 --w_f=0.01 

## 4. Citation

    @article{liu2023generalized,
        title={On Generalized Degree Fairness in Graph Neural Networks},
        author={Liu, Zemin and Nguyen, Trung-Kien and Fang, Yuan},
        journal={arXiv preprint arXiv:2302.03881},
        year={2023}
    }