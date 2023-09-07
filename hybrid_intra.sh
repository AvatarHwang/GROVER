NUM_NODES=1
NUM_GPU=4
WORLD_SIZE=$(($NUM_NODES * $NUM_GPU))
HOST_NODE_ADDR=192.168.100.72

torchrun \
--nnodes=$NUM_NODES \
--nproc_per_node=$NUM_GPU \
--node_rank=0 \
--master_addr=$HOST_NODE_ADDR \
--master_port=29526 \
HybridParallelGROVER.py
