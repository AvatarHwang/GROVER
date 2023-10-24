#!/bin/bash

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1              
#SBATCH --partition=gpu2
#SBATCH --exclude=n058
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=7
#SBATCH -o ./log/%j.sbatch.%N.out         
#SBATCH -e ./log/%j.sbatch.%N.err

#************************************************************
GRES="gpu:a10:1"
#************************************************************

mkdir -p ./log/$SLURM_JOB_ID

function get_master_adress(){
    NODE_LIST=`scontrol show hostnames $SLURM_JOB_NODELIST`
    MASTER_HOST=`echo $NODE_LIST | awk '{print $1}'`
    MASTER_ADDR=`cat /etc/hosts | grep $MASTER_HOST | awk '{print $1}'`
}
get_master_adress

SRUN_SCRIPT=$(cat <<EOF
    NODE_LIST=\$(scontrol show hostnames \$SLURM_JOB_NODELIST)
    node_array=(\$NODE_LIST)
    length=\${#node_array[@]}
    hostnode=\$(hostname -s)
    for ((i=0; i<\$length; i++)); do
        node=\${node_array[i]}
        if [ \$node == \$hostnode ]; then
            LOCAL_RANK=\$i
        fi
    done

    bash pretrain.sh \$LOCAL_RANK $MASTER_ADDR
EOF
)


srun --partition=$SLURM_JOB_PARTITION \
     --gres=$GRES \
     --cpus-per-task=7 \
     -o ./log/$SLURM_JOB_ID/%j.srun.%N.out \
     -e ./log/$SLURM_JOB_ID/%j.srun.%N.err \
     bash -c "$SRUN_SCRIPT"
