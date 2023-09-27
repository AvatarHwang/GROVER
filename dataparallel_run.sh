NUM_NODES=4
NUM_GPU_PER_NODE=1
NODE_RANK=$1
WORLD_SIZE=$(($NUM_NODES * $NUM_GPU_PER_NODE))
HOST_NODE_ADDR=192.168.120.86
BATCH_SIZE=512
DATA_PARALLEL_SIZE=$NUM_GPU_PER_NODE
MICRO_BATCH_SIZE=32
NUM_MICRO_BATCH=$(($BATCH_SIZE / $MICRO_BATCH_SIZE / $DATA_PARALLEL_SIZE))
MODEL_PARALLEL_SIZE=$NUM_NODES

torchrun \
--nnodes=$NUM_NODES \
--nproc_per_node=$NUM_GPU_PER_NODE \
--node_rank=$NODE_RANK \
--master_addr=$HOST_NODE_ADDR \
--master_port=29534 \
main.py finetune \
--data_path exampledata/finetune/toxcast.csv \
--features_path exampledata/finetune/toxcast.npz \
--save_dir model/finetune/toxcast/ \
--checkpoint_path model/grover_large.pt \
--dataset_type classification \
--split_type scaffold_balanced \
--ensemble_size 1 \
--num_folds 1 \
--no_features_scaling \
--epochs 10 \
--init_lr 0.00015 \
--ffn_hidden_size 700 \
--batch_size $BATCH_SIZE \
--world_size $WORLD_SIZE \
--pipeline_parallel \
--node_rank $NODE_RANK \
--num_micro_batch $NUM_MICRO_BATCH \
--micro_batch_size $MICRO_BATCH_SIZE \
--model_parallel_size $MODEL_PARALLEL_SIZE \
--data_parallel_size $DATA_PARALLEL_SIZE \
#--data_parallel \
#--task_parallel
