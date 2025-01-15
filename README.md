# Transfer_Random_forests_BBOB_Real_world

### Overview
This repository contains the implementation for the paper titled "Transfer Learning of Surrogate models via Domain Affine Transformation Across Synthetic and Real-World Benchmarks", which has been submitted to IEEE Congress on Evolutionary Computation.

### Installation
conda env create -f environment.yml

### Usage (for test on BBOB)
python TransferRF_BBOB/Riemannian_transferRF.py

### Usage (for test on real-world applications)
python TransferRF_Real_World/Riemannian_transferRF_asteroid_routing.py (for porkchop plot benchmarks in Interplanetary Trajectory Optimization)

python TransferRF_Real_World/Riemannian_transferRF.py (for Kinematics of a Robot Arm)

### Dataset
This repository contains the datasets we used for Porkchop Plot Benchmarks in Interplanetary Trajectory Optimization, Kinematics of a Robot Arm, and Single-Objective Game-Benchmark MarioGAN Suite. For the Real-World Optimization Benchmark in Vehicle Dynamics, please visit the original paper.