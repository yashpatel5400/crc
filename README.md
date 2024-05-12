# CRC (Conformalized Robust Control)
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
policy gradient method are the `policygradient.py` and `ltimult.py` scripts, which have the bulk of the implementations. Once the robust controllers
have been defined, they can be evaluated (to obtain the respective regrets) using
```
python evaluate.py --setup airfoil
```

## Extending CRC
To extend CRC, there are a number of directions of interest, as discussed in the paper. These include extensions to nonlinear dynamical
systems (using Koopman operators or neural operators) and applying the methods here to real engineering systems of interest in control co-design loops.
For these purposes, the main entrypoint for extension is `gen_data.py`: by adding a new synthetic environment or experimentally observed
trajectory data, a new system can directly be integrated into the workflow.