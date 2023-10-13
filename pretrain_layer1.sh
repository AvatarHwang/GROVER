NUM_NODES=1
NUM_GPU_PER_NODE=1
NODE_RANK=$1
WORLD_SIZE=$(($NUM_NODES * $NUM_GPU_PER_NODE))
HOST_NODE_ADDR=192.168.120.62
BATCH_SIZE=128
DATA_PARALLEL_SIZE=$NUM_GPU_PER_NODE
MICRO_BATCH_SIZE=128
NUM_MICRO_BATCH=$(($BATCH_SIZE / $MICRO_BATCH_SIZE / $DATA_PARALLEL_SIZE))
MODEL_PARALLEL_SIZE=$NUM_NODES

torchrun \
--nnodes=$NUM_NODES \
--nproc_per_node=$NUM_GPU_PER_NODE \
--node_rank=$NODE_RANK \
--master_addr=$HOST_NODE_ADDR \
--master_port=29599 \
main.py pretrain \
--data_path exampledata/pretrain/tryout \
--save_dir model/tryout \
--atom_vocab_path exampledata/pretrain/tryout_atom_vocab.pkl \
--bond_vocab_path exampledata/pretrain/tryout_bond_vocab.pkl \
--batch_size $BATCH_SIZE \
--dropout 0.1 \
--depth 5 \
--num_attn_head 1 \
--hidden_size 1200 \
--epochs 51 \
--warmup_epochs 2 \
--init_lr 0.00015 \
--max_lr 0.0003 \
--final_lr 0.000001 \
--weight_decay 0.0000001 \
--activation PReLU \
--world_size $WORLD_SIZE \
--backbone gtrans \
--embedding_output_type both \
--num_mt_block 4 \
--micro_batch_size $MICRO_BATCH_SIZE \
--data_parallel_size $DATA_PARALLEL_SIZE \
--num_micro_batch $NUM_MICRO_BATCH \
--model_parallel_size $MODEL_PARALLEL_SIZE \
--node_rank $NODE_RANK \
#--pipeline_parallel 