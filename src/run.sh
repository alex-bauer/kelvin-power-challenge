


# build features

cd features

echo "Merging Powerfiles"
python merge_power_files.py

echo "Creating DMOP"
python features_dmop_counts.py

echo "Creating EVTF"
python features_evtf_states.py

echo "Creating FTL"
#python features_ftl_states.py

echo "Creating LTDATA"
python features_ltdata.py

echo "Creating SAAF"
python features_saaf.py

cd ..

# run the model

cd modeling

python train_model.py





# 