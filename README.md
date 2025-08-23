# Unitree G1 Balance

MuJoCo environments, training scripts, and sim2real deployment code for the Unitree G1 EDU 23 DoF robot.

## Environment Setup

Clone the repository and change directories.

```bash
git clone git@github.com:mht3/unitree_g1_balance.git
cd unitree_g1_balance
```

Create a new conda environment with Python 3.10.
```bash
conda create -n g1_balance python=3.10
conda activate g1_balance
```

Install torch
<details>
<summary>PyTorch on GPU</summary>
<br>
Install PyTorch 2.7.1 with CUDA 12.6
  
```sh
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```
</details>

<details>
<summary>PyTorch on CPU Only</summary>
<br>
Alternatively, install PyTorch on the CPU.
  
```sh
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```
</details>

Install the remaining required packages.
```bash
pip install -r requirements.txt
```


Add the environment to ipykernel for running Jupyter notebooks.

```bash
python -m ipykernel install --user --name g1_balance --display-name "Python (g1_balance)"
```


### MuJoCo


To test that mujoco was correctly installed, try loading the Unitree G1 XML scene. A MuJoCo GUI should pop up.
```bash
python -m mujoco.viewer --mjcf=environments/unitree_robots/g1/scene_23dof.xml 
```

<img width="1275" height="756" alt="image" src="https://github.com/user-attachments/assets/1135f7b5-a2d5-4fbe-b352-81af07aeff72" />

## Real World Deployment


### Unitree Python SDK

Install `unitree_sdk2_python` as a package inside the `g11_balance` environment.

Initialize and update the `unitree_sdk2_python` submodule.

```bash
git submodule init
git submodule update
```

Set up the Unitree Python SDK environment. Assuming you have already cloned cyclonedds into the home directory:

```bash
cd unitree_sdk2_python
export CYCLONEDDS_HOME="~/cyclonedds/install"
pip3 install -e .
```

Refer to the unitree_sdk2_python README if you encounter any errors or have not installed cyclonedds.