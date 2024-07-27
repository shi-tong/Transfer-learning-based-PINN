# Transfer-learning-based-PINN
This repository contains the Python code for the paper "Prediction of 3D temperature field through single 2D temperature data based on transfer learning-based PINN model in laser-based directed energy deposition"

WARNING:
These codes are written only for the purpose of demonstration and verification. While the correctness has been carefully checked, the quality such as standardability, clarity, generality, and efficiency has not been well considered.

Note: The code is based on the previous study by Liao et al. [Liao, Shuheng, Tianju Xue, Jihoon Jeong, Samantha Webster, Kornel Ehmann, and Jian Cao. 2023. “Hybrid Thermal Modeling of Additive Manufacturing Processes Using Physics-Informed Neural Networks for Temperature Prediction and Parameter Identification.” Computational Mechanics 72 (3): 499–512. doi:10.1007/s00466-022-02257-9.]

In the present study, the heat conduction term and the convective heat transfer term are integrated to form a mixed heat conduction PDE to describe the temporal and spatial variation of the temperature field.
The PINN training process is based on transfer learning as follows:
1. Set the manufacturing and process parameters according to the source task, then train the model using the dataset from the source task (.../data / Source task /data.npy).
2. Basic model code path.(.../SourcePINN/PINN1/src/main.py).
3. Store the weights obtained from training on the source task as pre-training weights. (.../TargetPINN/PINN2/src/main.py).
4. Adjust the manufacturing and process parameters according to the target task, add the pre-training weights, and then use the dataset from the target task to train and fine-tune the model (.../data/Target task/data.npy).
5. Obtain the final training weights, total loss, PDE loss, BC loss, IC loss, and data-based loss(.../TargetPINN/TransformerPINN/src/main.py).
6. Combine the model and the preserved transfer learning-based training weights, input the spatio-temporal coordinates (x, y, z, t), and predict the temperature.
