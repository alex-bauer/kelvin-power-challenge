# kelvin-power-challenge

4th place solution of [Kelvin Mars Power Challenge](https://kelvins.esa.int/mars-express-power-challenge/).

### Instructions

#### Step 1. Install Dependencies
- Python 2.7
- Numpy
- Pandas 0.18 (for older verisons you need to change .resample() methods to old syntax)
- Scikit-learn 0.17
- Keras 1.04
- Xgboost 0.4

A convenient way to get all prerequisites is to install the [Anaconda](https://www.continuum.io/downloads) Python distribution. Use the script in env folder for automated download and install of anaconda dependencies.

####Step 2. Download source and competition data

1. Clone the repository using git: `git clone https://github.com/alex-bauer/kelvin-power-challenge.git`
2. Download data file from [competition website](https://kelvins.esa.int/mars-express-power-challenge/data/) and unpack to data folder

####Step 3. Best single models: Generate features and predict using single models (Xgboost, LSTM, DNN, ETR)

3. Run generate_features.sh
4. Run generate_single_models.sh

####Step 3. Generate model library and build ensemble using Caruana's greedy ensemble selection methods

5. Run build_ensemble.sh (this can take several days to finish, depending on your hardware)

#### Baseline solutions

Mean baseline:
Run `python mean_baseline.py`

Random Forest baseline:
4. Run `python rf_baseline.py` (0.12 on public leaderboard)

