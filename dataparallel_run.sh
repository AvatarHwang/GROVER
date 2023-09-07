NUM_NODES=2
NUM_GPU_PER_NODE=4
NODE_RANK=$1
WORLD_SIZE=$(($NUM_NODES * $NUM_GPU_PER_NODE))
HOST_NODE_ADDR=192.168.120.93
BATCH_SIZE=32

torchrun \
--nnodes=$NUM_NODES \
--nproc_per_node=$NUM_GPU_PER_NODE \
--node_rank=$NODE_RANK \
--master_addr=$HOST_NODE_ADDR \
--master_port=29526 \
main.py finetune \
--data_path exampledata/finetune/tox21.csv \
--features_path exampledata/finetune/tox21.npz \
--save_dir model/finetune/tox21/ \
--checkpoint_path model/grover_large.pt \
--dataset_type classification \
--split_type scaffold_balanced \
--ensemble_size 1 \
--num_folds 1 \
--no_features_scaling \
--epochs 2 \
--init_lr 0.00015 \
--ffn_hidden_size 700 \
--batch_size $BATCH_SIZE \
--data_parallel \
--world_size $WORLD_SIZE
