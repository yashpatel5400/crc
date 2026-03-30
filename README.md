<h1 align='center'>Conformal Robust Control of Linear Systems</h1>

<div align='center'>
    <a href='https://ypatel.io/' target='_blank'>Yash Patel</a><sup>1</sup>&emsp;
    <a href='https://srayan00.github.io/' target='_blank'>Sahana Rayan</a>&emsp;;
    <a href='https://www.ambujtewari.com/' target='_blank'>Ambuj Tewari</a><sup>2</sup>&emsp;
</div>

<div align='center'>
Department of Statistics, University of Michigan.
</div>

<p align='center'>
    <sup>2</sup>Senior investigator
</p>
<div align='center'>
    <a href='https://arxiv.org/abs/2405.16250'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://openreview.net/attachment?id=rdsOb7q6mw&name=pdf'><img src='https://img.shields.io/badge/Paper-AISTATS026-blue'></a>
</div>

## Instructions
The code is primarily broken into the following files and builds upon the tremendous work in https://github.com/TSummersLab/polgrad-multinoise:
- `gen_data.py`: Generates synthetic trajectories under LTI systems with random gain matrices; extracts estimates with DMDc
- `train.py`: Train a contextual LTI predictor model; conformalize against a held-out subset of the observations
- `crc.py`: Runs a modified policy gradient to obtain the robust controller under the conformal prediction region

## Reproducing results
To reproduce results, we have two setups supported: airfoil and load_pos. To reproduce the airfoil results, we first generate data with:
```
python generate_data.py --setup airfoil
```
This will produce a `data/airfoil` directory with training, calibration, and test data. With this, we can train and calibrate a model using:
```
python train.py --setup airfoil
```
This will produce a file in `experiments/` ready for ingestion into the robust control pipeline with predicted dynamics and a quantile. We also save the true dynamics
to enable final evaluation of the robust controller (i.e. evaluating the regret). To run the control algorithm, run:
```
python crc.py --setup airfoil
```
This will produce a file in the `results/` directory with controllers defined using CRC and alternate margin methods. The main workhorses in this
policy gradient method are the `policygradient.py` and `ltimult.py` scripts in `polgrad/`, which have the bulk of the implementations. Once the robust controllers
have been defined, they can be evaluated (to obtain the respective regrets) using
```
python evaluate.py --setup airfoil
```

## ⚖️ Disclaimer
This project is intended for academic research, and we explicitly disclaim any responsibility for user-generated content. Users are solely liable for their actions while using the generative model. The project contributors have no legal affiliation with, nor accountability for, users' behaviors. It is imperative to use the generative model responsibly, adhering to both ethical and legal standards.

## &#x1F4D2; Citation

If you find our work useful for your research, please consider citing the paper :

```
@article{patel2024conformal,
  title={Conformal robust control of linear systems},
  author={Patel, Yash and Rayan, Sahana and Tewari, Ambuj},
  journal={arXiv preprint arXiv:2405.16250},
  year={2024}
}
```
