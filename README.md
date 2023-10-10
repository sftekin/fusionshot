# Installation
```
$ pip install requirements.txt
```
# Running

## Obtaining Predictions from Base Models
run
```
$ cd base_model_src/
$ ./inference.sh
```

## Pruning the ensemble set

```
$ cd ens_pruning_src/
$ ./prune.sh
```

## Training and Running Fusionshot

```
$ cd fusionshot_src/
$ python train_ensemble.py
```

