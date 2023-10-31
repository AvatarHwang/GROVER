# Efficient Model Parallelism for GROVER

This repository deals with model parallelism, Data+Task Model Parallelism and Task+Pipeline model parallelism for GROVER. To see what GROVER is, please refer [GROVER's original GitHub repository](https://github.com/tencent-ailab/grover).

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

## Installation

To install the repository, follow the steps below: 
1.  clone the repository
```bash
~$git clone git@github.com:AvatarHwang/TaskModelParallelism.git
```

The requirements follow the original github of GROVER: [GROVER's original GitHub repository](https://github.com/tencent-ailab/grover)



## Usage

### Data+Task Model Parallelism

Task model parallism for GROVER is dividing the GROVER into two sub-model, Node Encoder and Edge Encoder. To use this parallelization strategy, you'll need at least 2 GPUs. To use combined data+task parallelism, you will need at least two GPU nodes, and each node has to have more than 1 GPU. 

To run Data+Task Model Parallelism, follow the steps below:
1. Config `GROVER/hybrid_run_inter_node.sh` and `GROVER/hybrid_run_inter_edge.sh`
```bash
NUM_NODES=2
NUM_GPU_PER_NODE=(TYPE THE NUMBER OF GPU ON YOUR NODE)
NODE_RANK=$1
WORLD_SIZE=$(($NUM_NODES * $NUM_GPU_PER_NODE))
HOST_NODE_ADDR=(TYPE THE IP ADDRESS OF YOUR MASTER NODE)

torchrun \
--nnodes=$NUM_NODES \
--nproc_per_node=$NUM_GPU_PER_NODE \
--node_rank=$NODE_RANK \
--master_addr=$HOST_NODE_ADDR \
--master_port=29525 \
HybridParallel_Internode_node.py
```

2. Run `hybrid_run_inter_node.sh` on a node.
```bash
$bash hybrid_run_inter_node.sh 0
```

3. Run `hybrid_run_inter_edge.sh` on another node.
```bash
$bash hybrid_run_inter_edge.sh 1
```

### Task+Pipeline Model Parallelism

Task+Pipeline model parallelism deals with GROVER with more GTransformer layers than the original GROVER. Adding more layers potentially has some benefits on model accuracy. Basically, this is only for Pretraining the model. This parallelization strategy divide GROVER with more layers into GROVER with one layer. 

To run Task+Pipeline Model Parallelism, follow the steps below:
1. Config `GROVER/pretrain.sh`. 

```bash
NUM_NODES=(NUMBER OF GPUs)
NUM_GPU_PER_NODE=1
NODE_RANK=$1
WORLD_SIZE=$(($NUM_NODES * $NUM_GPU_PER_NODE))
HOST_NODE_ADDR=$2
BATCH_SIZE=$((32 * $NUM_NODES))
DATA_PARALLEL_SIZE=$NUM_GPU_PER_NODE
MICRO_BATCH_SIZE=32
NUM_MICRO_BATCH=$(($BATCH_SIZE / $MICRO_BATCH_SIZE / $DATA_PARALLEL_SIZE))
MODEL_PARALLEL_SIZE=$NUM_NODES

#echo "Master Address: $HOST_NODE_ADDR"

#source ~/miniconda3/bin/activate grover

torchrun \
--nnodes=$NUM_NODES \
...
--pipeline_parallel
```

2. Run the shell script on each GPU. The {GPU Rank: for example, 0} should be replaced by the user with the appropriate GPU rank.
```bash
$bash pretrain.sh {GPU Rank: for example, 0}
```

## Citation

If you use our work, please cite it as:

1. Data+Task Model Parallelism:

@article{황순열2022grover,
  title={GROVER 의 모델 병렬적 및 데이터 병렬적 딥러닝},
  author={황순열 and 이은경 and 이영민},
  journal={한국정보과학회 학술발표논문집},
  pages={1157--1159},
  year={2022}
}

2. Task+Pipeline Model Parallelism: (will be added soon)


## License

This software is released under the MIT License, with the additional requirement that any use of the software, or derivatives of it, in any form, must cite the associated work or the software repository itself.

### MIT License

Copyright (c) 2023 Sunyeol Hwang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
