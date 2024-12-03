## Environment Setup:
````
conda create -n haochi-env python=3.11
conda activate haochi-env
pip install -r requirements.txt
````

## Train prompt:
````
cd training_scripts
python train.py \
  --root_dir ls ../data_scripts/dataset/ChineseFoodNetDataset/all/ \
  --train_csv ../data_scripts/dataset/csv/train_data.csv \
  --val_csv ../data_scripts/dataset/csv/test_data.csv \
  --results_dir ../results \
  --wandb_key <your_wandb_api_key> \
  --batch_size 32 \
  --epochs 100 \
  --lr 0.0001
````
