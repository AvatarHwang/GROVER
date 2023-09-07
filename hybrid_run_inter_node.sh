NUM_NODES=2
NUM_GPU_PER_NODE=4
NODE_RANK=$1
WORLD_SIZE=$(($NUM_NODES * $NUM_GPU_PER_NODE))
HOST_NODE_ADDR=192.168.120.95

torchrun \
--nnodes=$NUM_NODES \
--nproc_per_node=$NUM_GPU_PER_NODE \
--node_rank=$NODE_RANK \
--master_addr=$HOST_NODE_ADDR \
--master_port=29525 \
HybridParallel_Internode_node.py