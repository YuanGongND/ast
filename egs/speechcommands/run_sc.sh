#!/bin/bash       

#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="aed-sc"
#SBATCH --output=./slurm_logs/esc-%j.txt

set -x
. /data/sls/scratch/share-201907/slstoolchainrc
base_dir=/data/sls/scratch/yuangong/audioset
source /data/sls/scratch/yuangong/aed-trans/venv-trans/bin/activate
export TORCH_HOME=/data/sls/scratch/yuangong/aed-trans/model/

effmode=trans
subset=speechcommand
bal=none
lr=0.00001
patience=2
freqm=48
timem=48
mixup=0.6
freeze=True
batchsize=128
mdlversion=8
exp_dir=/data/sls/scratch/yuangong/aed-trans/exp/testspeechcommand22-$lr-$freeze-v$mdlversion-mix${mixup}-fm$freqm-tm$timem-bs$batchsize-correctroll20-20true-valid-truenewschedule-flexlr

if [ -d $exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore ../run_sc.py --lr $lr --data-train $base_dir/datafiles/speechcommand_train_data.json \
--data-val $base_dir/datafiles/speechcommand_valid_data.json --exp-dir $exp_dir --clean-start --train-mode \
--n-print-steps 10 --num-workers 8 --label-csv $base_dir/utilities/speechcommand_class_labels_indices.csv --n_class 35 --n-epochs 30 --batch-size $batchsize \
--apc_trainable $freeze --apc_rnn_layer 2 --pretrain_mode ${effmode} --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} --lr_patience ${patience} --mdlversion ${mdlversion}
