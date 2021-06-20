#!/bin/bash
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="aed-esc"
#SBATCH --output=./slurm_logs/esc-%j.txt

set -x
. /data/sls/scratch/share-201907/slstoolchainrc
base_dir=/data/sls/scratch/yuangong/audioset
source /data/sls/scratch/yuangong/aed-trans/venv-trans/bin/activate
export TORCH_HOME=/data/sls/scratch/yuangong/aed-trans/model/

effmode=trans
subset=esc50
bal=none
lr=0.0001
patience=2
freqm=24
timem=96
mixup=0
freeze=False
mdlversion=9
base_exp_dir=/data/sls/scratch/yuangong/aed-trans/exp/testesc16-$lr-$freeze-v$mdlversion-mix${mixup}-fm$freqm-tm$timem-cehard-newschedule-bettermodel-repeat3

if [ -d $base_exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p exp_dir

for((fold=1;fold<=5;fold++));
do
  echo 'now process fold'${fold}

  exp_dir=${base_exp_dir}/fold${fold}
  rm -rf exp_dir
  mkdir -p exp_dir

  CUDA_CACHE_DISABLE=1 python -W ignore ../run_esc.py --lr $lr --data-train $base_dir/datafiles/esc_train_data_${fold}.json \
  --data-val $base_dir/datafiles/esc_eval_data_${fold}.json --exp-dir $exp_dir --clean-start --train-mode \
  --n-print-steps 100 --num-workers 8 --label-csv $base_dir/utilities/esc_class_labels_indices_pro.csv --n_class 50 --n-epochs 35 --batch-size 48 \
  --apc_trainable $freeze --apc_rnn_layer 2 --pretrain_mode ${effmode} --save_model True \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} --lr_patience ${patience} --mdlversion ${mdlversion}
done

python ../get_esc_result.py --exp_path ${base_exp_dir}
python ../get_esc_summary.py
