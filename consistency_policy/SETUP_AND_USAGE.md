# Tips for setup if you prefer pip, use Ubuntu22, and have an NvidiaRTX 3090

In general, you should probably read the setup instructions in the [original README](../README.md). For my preferred setup, I needed to do a little more. Here's how I got this code working for me.

### Python version

Use Python3.8. I would have liked to use 3.10 but there are too many compatibility issues.

### CUDA version

I'm not 100% sure this is needed. Try not doing this and come back later if you need to (PyTorch3D might complain).

Install CUDA 12.1. This matches up with the version of CUDA that PyTorch 2.1.0 ships with. You can follow the instructions here https://developer.nvidia.com/cuda-12-1-1-download-archive.

### Mujoco

Install Mujoco 2.1.0.

1. Download it from here https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz.
2. Unzip and put it at `~/.mujoco/mujuco210`. The `bin` folder should be a subdirectory of this, so do what you need to do to get the level of nesting right.
3. Add this to your bashrc or equivalent `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/alexander/.mujoco/mujoco210/bin`.
4. You may need to put the license key in `~/.mujoco/mjkey.txt` (I can't remember if it's needed for this version, probably not). Get it from here https://www.roboti.us/license.html.

Also take note of the comment in `conda_environment.yaml`. For the pip package free-mujoco-py it says: `requires mujoco py dependencies libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf`

### The bulk of the packages

This closely follows the package versions in conda_environment.yaml but might have some additional packages that I use (deepdiff, timm and maybe more).

`pip install -r consistency_policy/requirements.txt`

### PyTorch3D
 
Fingers crossed this works for you. I had issues with matching up CUDA version and GCC version but this one did the trick. The build takes a while.

`pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"`

### Gym

There's an issue when trying to install 0.21.0. Following [this](https://github.com/openai/gym/issues/3202), I made my own fork:

```
pip install git+https://github.com/alexander-soare/gym.git@156cc2c90f3c1eced9e57773a7dfc67e341cf120
```

### Robosuite

This particular version is needed for some of the robot environments.

`pip install robosuite@https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz`

# Usage

## Training

To train a consistency model, you can hand-tweak the configs in `consistency_policy/train_configs`. Here's how I've set it up: I copy pasted the configs from the original work's [data store](https://diffusion-policy.cs.columbia.edu/data/) and named them starting with `diffusion_`. Then for the consistency policy version, I made a corresponding `consistency_` config and overrode/added config parameters where necessary. The configs as they are now, are what I used to produce good results.

So when you're happy with your config just run: `python -m consistency_policy.train ----config-name=PATH/TO/YAML`.

Note that training and eval won't work for all environments and tasks. It will just work for (at least) what I've included in the configs. You'll probably have to do some tweaking if you want more.

## Eval

You can run `python -m consistency_policy.eval --checkpoint PATH/TO/CKPT --eval-step-size 5` to evaluate a single model with a set step size for inference.