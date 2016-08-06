
# run the modeling

cd modeling

python train_random_subset_models.py f12_xgb
python train_random_subset_models.py f12_lstm

python hillclimb.py