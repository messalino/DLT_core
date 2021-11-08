Variables in dlt_config.ini:
n_ts_in (integer): number of time steps of the time series that are used to perform training and prediction
n_features (integer): number of features given as input to the LSTM
model_name (string): name of the folder containing the LSTM model
model_dir (path): data/models
metrics_dir (path): data/metrics
load_model (boolean): wheter to load a pre-existing model or create a new one
batch_size (integer): size of each batch used to train LSTM models
n_epochs (integer): number of epochs used to train LSTM models
n_ts_failure (integer): number of seconds corresponding to the time window within which the failure is predicted
time_step (integer): number of seconds corresponding to the interval between two consecutive measurements