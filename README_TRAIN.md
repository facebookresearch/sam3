# Training

This repository supports finetuning SAM3 models on custom datasets in multi-node setup or local execution. The training script is located at `sam3/train.py` and uses Hydra configuration management to handle complex training setups.


## Installation

```bash
cd sam3
pip install -e ".[train]"
```

### Training Script Usage

The main training script is located at `sam3/train.py`. It uses Hydra configuration management to handle complex training setups.

#### Basic Usage

```bash
# Example: Train on Roboflow dataset
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml
# Example: Train on ODinW13 dataset
python sam3/train/train.py -c configs/odinw13/odinw_text_only_train.yaml
```
Follow [`Roboflow 100-VL`](https://github.com/roboflow/rf100-vl/) to download the roboflow 100-vl datasets. Follow [`GLIP`](https://github.com/microsoft/GLIP) to download the ODinW datasets. The data folder should be organized as follows, and put your roboflow_vl_100_root and odinw_data_root in the job configs.
```
roboflow_vl_100_root:
  13-lkc01
    train
    valid
    test
  2024-frc
  actions
  ...
odinw_data_root:
  AerialMaritimeDrone
    large
      train
      valid
      test
  Aquarium
  ...
```

#### Command Line Arguments

The training script supports several command line arguments:

```bash
python sam3/train/train.py \
    -c CONFIG_NAME \
    [--use-cluster 0|1] \
    [--partition PARTITION_NAME] \
    [--account ACCOUNT_NAME] \
    [--qos QOS_NAME] \
    [--num-gpus NUM_GPUS] \
    [--num-nodes NUM_NODES]
```

**Arguments:**
- `-c, --config`: **Required.** Path to the configuration file (e.g., `sam3/train/configs/roboflow_v100_full_ft_100_images.yaml`)
- `--use-cluster`: Whether to launch on a cluster (0: local, 1: cluster). Default: uses config setting
- `--partition`: SLURM partition name for cluster execution
- `--account`: SLURM account name for cluster execution
- `--qos`: SLURM QOS (Quality of Service) setting
- `--num-gpus`: Number of GPUs per node. Default: uses config setting
- `--num-nodes`: Number of nodes for distributed training. Default: uses config setting

#### Local Training Examples

```bash
# Single GPU training
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0 --num-gpus 1

# Multi-GPU training on a single node
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0 --num-gpus 4

# Force local execution even if config specifies GPUs
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0
```

#### Cluster Training Examples

```bash
# Basic cluster training with default settings from config
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 1

# Cluster training with specific SLURM settings
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml \
    --use-cluster 1 \
    --partition gpu_partition \
    --account my_account \
    --qos high_priority \
    --num-gpus 8 \
    --num-nodes 2
```

### Configuration Files

Training configurations are stored in `sam3/train/configs/`. The configuration files use Hydra's YAML format and support:

- **Dataset Configuration**: Data paths, transforms, and loading parameters
- **Model Configuration**: Architecture settings, checkpoint paths, and model parameters
- **Training Configuration**: Batch sizes, learning rates, optimization settings
- **Launcher Configuration**: Distributed training and cluster settings
- **Logging Configuration**: TensorBoard, experiment tracking, and output directories

#### Key Configuration Sections

```yaml
# Paths to datasets and checkpoints
paths:
  bpe_path: /path/to/bpe/file
  dataset_root: /path/to/dataset
  experiment_log_dir: /path/to/logs

# Launcher settings for local/cluster execution
launcher:
  num_nodes: 1
  gpus_per_node: 2
  experiment_log_dir: ${paths.experiment_log_dir}

# Cluster execution settings
submitit:
  use_cluster: True
  timeout_hour: 72
  cpus_per_task: 10
  partition: null
  account: null
```

### Monitoring Training

The training script automatically sets up logging and saves outputs to the experiment directory:

```bash
# Logs are saved to the experiment_log_dir specified in config
experiment_log_dir/
├── config.yaml              # Original configuration
├── config_resolved.yaml     # Resolved configuration with all variables expanded
├── checkpoints/             # Model checkpoints (if skip_checkpointing=False)
├── tensorboard/             # TensorBoard logs
├── logs/                    # Text logs
└── submitit_logs/           # Cluster job logs (if using cluster)
```

You can monitor training progress using TensorBoard:

```bash
tensorboard --logdir /path/to/experiment_log_dir/tensorboard
```

### Job Arrays for Dataset Sweeps

The Roboflow and ODinW configuration supports job arrays for training multiple models on different datasets:

This feature is specifically enabled via,
```yaml
submitit:
  job_array:
    num_tasks: 100
    task_index: 0
```

The configuration includes a complete list of 100 Roboflow supercategories, and the `submitit.job_array.task_index` automatically selects which dataset to use based on the array job index.

```bash
# Submit job array to train on different Roboflow datasets
# The job array index selects which dataset from all_roboflow_supercategories
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml \
    --use-cluster 1
```

### Reproduce ODinW13 10-shot results
Running the following job will give the results on the ODinW13 seed 300, see `odinw_train.train_file: fewshot_train_shot10_seed300` in the config file.
```bash
# Example: Train on ODinW13 dataset
python sam3/train/train.py -c configs/odinw13/odinw_text_only_train.yaml
```
Change `odinw_train.train_file` to `fewshot_train_shot10_seed30` and `fewshot_train_shot10_seed3` to get the results for the other two seeds. Final results are aggregated from the three seeds. Notice that a small number of jobs may diverge during training, in which case we just use the last checkpoint's result before it diverges.


### Eval Script Usage
With a similar setup as the training config, the training script `sam3/train.py` can also be used for evaluation, too, when setting `trainer.mode = val` in the job config. Run the following job will give the results on the zero-shot results on RF100-VL and ODinW13 datasets.
```bash
# Example: Evaluate on Roboflow dataset
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_eval.yaml
# Example: Evaluate on ODinW13 dataset
python sam3/train/train.py -c configs/odinw13/odinw_text_only.yaml
```
