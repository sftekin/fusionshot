ep_num=600
seed=50
dataset_name="miniImagenet"
n_shot=1

# CloserLook Models
for method in "relationnet" "protonet" "matchingnet" "maml_approx"
do
  for model_name in 'Conv4' 'Conv6' 'ResNet10' 'ResNet18' 'ResNet34'
  do
    for data_set in "base" "val" "novel"
    do
        python inference.py --method $method --model_name $model_name --class_type $data_set --dataset_name $dataset_name --n_shot $n_shot --ep_num $ep_num --seed $seed
    done
  done
done

# Simple shot Models
for model_name in 'Conv4' 'Conv6' 'ResNet10' 'ResNet18' 'ResNet34' 'DenseNet121' 'WideRes'
do
  for data_set in "base" "val" "novel"
  do
      python inference.py --method simpleshot --class_type $data_set --dataset_name $dataset_name --n_shot $n_shot --ep_num $ep_num --seed $seed --model_name $model_name
  done
done

# DeepEMD
for data_set in "base" "val" "novel"
do
  python inference.py --method DeepEMD --class_type $data_set --dataset_name $dataset_name --n_shot $n_shot --ep_num $ep_num --seed $seed
done
