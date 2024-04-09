ep_num=300
seed=50
dataset_name="CUB"
n_shot=5

# CloserLook Models
for method in "matchingnet"
do
  for model_name in 'ResNet18'
  do
    for data_set in "base" "val" "novel"
    do
        python inference.py --method $method --model_name $model_name --class_type $data_set --dataset_name $dataset_name --n_shot $n_shot --ep_num $ep_num --seed $seed
    done
  done
done

## Simple shot Models
#for model_name in 'WideRes'
#do
#  for data_set in "base" "val" "novel"
#  do
#      python inference.py --method simpleshot --class_type $data_set --dataset_name $dataset_name --n_shot $n_shot --ep_num $ep_num --seed $seed --model_name $model_name
#  done
#done
#
## DeepEMD
#for data_set in "base" "val" "novel"
#do
#  python inference.py --method DeepEMD --class_type $data_set --dataset_name $dataset_name --n_shot $n_shot --ep_num $ep_num --seed $seed
#done
##
##
##
## EASY
#for data_set in "base" "val" "novel"
#do
#  python inference.py --method easy --class_type $data_set --dataset_name $dataset_name --n_shot $n_shot --ep_num $ep_num --seed $seed
#done
