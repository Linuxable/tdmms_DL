
# How-to-DelftBlue
This documentation explains how to setup and use the supercomputer DelftBlue. The official documentation can be found [here](https://doc.dhpc.tudelft.nl/delftblue/). They also have a [crash-course for beginners](https://doc.dhpc.tudelft.nl/delftblue/crash-course/). A lot of things are already explained very well in the official docs, so I will use that and link certain pages.

DelftBlue consists of login and computation nodes. The login nodes are **not** meant to run programs. That is where the computations nodes are for. More on this later.

DelftBlue is a Linux machine, so you can use Linux commands to navigate the login node.

**Access**
The first step is to get access to DelftBlue, you can request access from [this page](https://doc.dhpc.tudelft.nl/delftblue/Guest-Access/).
Please note which type of account you received. For example `research-as-bn` or `research-as-qn`. More information can be found [here](https://doc.dhpc.tudelft.nl/delftblue/Accounting-and-shares/).

**Login**
You can login to DelftBlue through a terminal using SSH:
`$ ssh <netid>@login.delftblue.tudelft.nl`

This will log you into one of the four login nodes.
Remember to use the TU Delft internet or EduVPN.

> See [this page](https://doc.dhpc.tudelft.nl/delftblue/Remote-access-to-DelftBlue/) for more access methods. Such as [MobaXterm](https://mobaxterm.mobatek.net/), which has a GUI.

To make life a little easier we can configure SSH keys, so we never have to enter our TU Delft password.
The SSH configuration file can be found in `~/.ssh/config` on Linux, or in `C:\Users\<username>\.ssh` on Windows.
Place the following lines in the configuration file:
```
Host delftblue
User <netid>
HostName login.delftblue.tudelft.nl
Port 22
```
Now we can connect to DelftBlue be running:
`$ ssh delftblue`
 

To generate the SSH keys run:
`$ ssh-keygen`
When asked for a passphrase hit enter.

Copy the keys to DelftBlue:
`$ ssh-copy-id delftblue` or `$ ssh-copy-id <netid>@login.delftblue.tudelft.nl`

**Directory**
You have two directories in DelftBlue. The login and computation nodes have access to these directories.
-  `/home/<netid>`: limit of 30GB. Permanent storage.
-  `/scratch/<netid>`: limit 5TB. Temporary storage, gets cleaned every 4 - 6 months. **Not backed up.**

We will make our Conda environments in the `scratch/` directory and keep our scripts and configuration files in the `home/` directory.

Create the following two folders in your `home/` directory, `stdouts/` and `stderrs/`.
This can be done by running the following from you `home/` directory:
`$ mkdir <folder name>`

These folders will be used for the outputs of your programs.

When logging in a message shows the current memory usage of `home/` and `scratch/`.  You can also invoke this message by running `$ exec bash`. If the `home/` directory reaches `30GB` you should clean up. Otherwise code will not be able to run and you will not be able to upload data to DB. 
>An `[Errno 112] Disk quota exceeded` error, resulting from for example a full `home/` directory will not be logged in the `stderrs/` file.

**Transfer Data**
*Files and folders*
To transfer files and folders to DelftBlue using `scp` from the terminal please see [this page](https://doc.dhpc.tudelft.nl/delftblue/Data-transfer-to-DelftBlue/). This can also be down with a GUI, by using MobaXterm.

*Code*
Of course it is also possible to import our code into DelftBlue using `scp` or through dragging and dropping files. But I find it easier to work with a Github repository, which already has many benefits for project and version management.

To import your Github repository run:
`$ git clone <repo url>`

And to update the code, `cd` tot the repository and run:
`$ git pull`

**Setup Conda**
Before doing anything with Conda, run the following commands.
`$ module load miniconda3`
`$ mkdir -p /scratch/${USER}/.conda`
`$ ln -s /scratch/${USER}/.conda $HOME/.conda`

This will store the Conda environments in the `scratch/` directory, since they can eat up a lot of space.
It is possible to create Conda environments from the login node. Later will be discussed how to setup Conda for GPU usage.

**Use Software Modules**
Instead of using Conda for python packages it is also possible to load the packages as software modules. More information can be found [here](https://doc.dhpc.tudelft.nl/delftblue/DHPC-modules/). For example:
```
#!/bin/sh

#SBATCH --job-name="tdmms_dl-training"
#SBATCH -e stderrs/%j-tdmms_dl-training.err
#SBATCH -o stdouts/%j-tdmms_dl-training.out
#SBATCH --partition=gpu-a100
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --account=education-as-bsc-tn

module load 2023r1
module load py-numpy/1.26.1
module load py-tensorflow/2.16.1

srun python train.py
```

To see all available modules run:
`$ module avail`
To get specific information of a module run:
`$ module spider <module> `
To search for modules matching keywords run:
`$ module keyword <key1> <key2> ...`

**Run Jobs**
DelftBlue uses a program called Slurm to manage all the different users and computation nodes. To run our code we need to submit a batch job. This is a batch script that contains all the information about the job and which file to run.

To submit a batch job run:
`$ sbatch batchjob.sh`
>Note that the command `sbatch` must always be ran from the `home/` directory. To go to the home directory you can run `$ cd ~/`.
More information about batch jobs can be found [here](https://doc.dhpc.tudelft.nl/delftblue/Slurm-scheduler/).

Below is an example `batchjob.sh` file.
```
#!/bin/sh

#SBATCH --job-name="tdmms_dl-training"
#SBATCH -e stderrs/%j-tdmms_dl-training.err
#SBATCH -o stdouts/%j-tdmms_dl-training.out
#SBATCH --partition=gpu-a100
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --account=education-as-bsc-tn

module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate env_tf24

previous=$(nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')
nvidia-smi

cd /home/aldelange/ai/tdmms_DL
srun python train.py

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

conda deactivate
```
Explanation
```
#!/bin/sh

#SBATCH --job-name="tdmms_dl-training"
#SBATCH -e stderrs/%j-tdmms_dl-training.err
#SBATCH -o stdouts/%j-tdmms_dl-training.out
#SBATCH --partition=gpu-a100
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --account=education-as-bsc-tn
```
This part states what the output files of the program should be. Python `print` statements will for example be written in the `.out` file, and errors will be written in the `.err` file. Feel free to rename the `job-name` , `.out` and `.err` files.

It also states which partition to use, what the run time is, how many CPUs and GPUs you want and how much RAM you want to use.
>Note that the more recourses you request, the longer you have to wait for your code to run. Slurm generally gives small and shorter programs a shorter waiting time. If you want to test things on the GPUs and do not need a lot of memory, use the `gpu-a100-small` partition.

**Remember** to enter your own account type after `--account`.

There are multiple hard stops of your code. Two of them are:
1. Runtime, `--time`, if this time is up your code will be stopped.
2. Memory, `--mem-per-cpu`, if your program uses to much memory it will also directly be stopped.

```
module load miniconda3
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env_tf24
```
`conda deactivate`

These lines are to activate and deactivate your Conda environment.
```
previous=$(nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

nvidia-smi
```

```
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
```

These lines are to see how much you used of the requested GPUs. This will print two lines in the `.out` file when the program finishes.
```
cd /home/aldelange/ai/tdmms_DL

srun python train.py
```
These lines are to run the program.

**Check Jobs**
To see the status of your job run:
`$ squeue --me`
With a nicer format:
`$ squeue --format="%.18i %.9P %.90j %.8u %.8T %.10M %.9l %.6D %R" --me`
To get an estimation of when your job will run:
`$ squeue --me --start`
To cancel a job run:
`$ scancel <job id>`
See recent run jobs:
`$ sacct`

**Conda For GPU Usage with Tensorflow or PyTorch**
To setup Conda for GPU usage we will create the Conda environment on a GPU node. This is done for example by the following batch job (for Tensorflow):
```
#!/bin/sh
#SBATCH --job-name="tdmms_dl-training"
#SBATCH -e stderrs/%j-tdmms_dl-training.err
#SBATCH -o stdouts/%j-tdmms_dl-training.out
#SBATCH --partition=gpu-a100-small
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --account=education-as-bsc-tn

module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n env_tf24 python=3.8.10 -c conda-forge
conda activate env_tf24

conda install cudatoolkit=11.0 -c conda-forge
conda install cudnn=8.0 -c conda-forge

cd /home/aldelange/ai/tdmms_DL
pip install -r requirements.txt

conda deactivate
```
For Tensorflow check [this page](https://www.tensorflow.org/install/source#gpu) for Cuda compatibility. It is important to first install the correct `cudatoolkit` and `cudnn` versions. And then afterwards to install Tensorflow **using**  `pip`, not Conda (this is done in the batch job through the `requirements.txt` file). Because `conda install tensorflow` will install the wrong `cudatoolkit` and `cudnn` versions.
> The output files from Tensorflow during training, TensorBoard files and model weights checkpoints, quickly use a lot of memory. 

For PyTorch please see [this page](https://pytorch.org/get-started/locally/). It will generate an command that will install everything in one go. This will result in something like this:
```
#!/bin/sh
#SBATCH --job-name="tdmms_dl-training"
#SBATCH -e stderrs/%j-tdmms_dl-training.err
#SBATCH -o stdouts/%j-tdmms_dl-training.out
#SBATCH --partition=gpu-a100-small
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --account=education-as-bsc-tn
  
module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
  
conda create -n env_tf24
conda activate env_tf24
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  
cd /home/aldelange/ai/tdmms_DL
pip install -r requirements.txt

conda deactivate
```
And now `pytorch` does not have to be installed using `pip`.
  
**Contact**
If you have any questions please contact Abel de Lange, a.l.delange@student.tudelft.nl.