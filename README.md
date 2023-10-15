# Installation
```
$ pip install requirements.txt
```

## Datasets
Download the mini-Imagenet and CUB datasets at the following link and extract them under `base_model_src/filelists/<datasetname>`

- Mini-Imagenet Link: https://www.dropbox.com/scl/fi/1i5zt3m3o1hmu34ywsl9d/miniImagenet.zip?rlkey=hlwhs3p729uyzpn5yd785lb7c&dl=0
- CUB Link: https://www.dropbox.com/scl/fi/j6208ndbc7e7qvzo0qfgh/CUB.zip?rlkey=3lcyfpmx85wa46u5bk814goki&dl=0

## Models
The trained base-models for each dataset can be found at the link on below.

- CUB models Link: https://www.dropbox.com/scl/fi/92ebuenjg4sqfb4z9oedm/CUB_models.zip?rlkey=90fo8gouds3zmsf5xuzmeqhm0&dl=0
- Mini-Imagenet models Link: https://www.dropbox.com/scl/fi/y0ythukn7zmm7fhqmzq6f/miniImagenet_models.zip?rlkey=5rq2sm8xbqj0vakbs96lpxwzr&dl=0

Extract them under `checkpoints/`.



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

