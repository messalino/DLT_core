import numpy as np
import configparser
import os

from lstm import LSTM

if __name__ == '__main__':

    # READ CONFIGURATION FILE
    config_file_path = "./dlt_config.ini"
    config = configparser.ConfigParser()
    try:
        config.read(config_file_path)
    except:
        print("Error: Could not read configuration file, exiting...")
        exit(1)

    # GET PARAMETERS FROM CONFIG FILE
    n_ts_in = config["lstm"].getint("n_ts_in")
    n_features = config["lstm"].getint("n_features")
    model_name = config["lstm"]["model_name"]
    model_dir = config["lstm"]["model_dir"]
    metrics_dir = config["lstm"]["metrics_dir"]

    load_model = config["dlt"].getboolean("load_model")
    batch_size = config["dlt"].getint("batch_size")
    n_epochs = config["dlt"].getint("n_epochs")
    n_ts_failure = config["dlt"].getint("n_ts_failure") # 120 (sec.) -> predict a failure in the next 20 minutes
 
    print("Parameters retrieved from configuration file")

    # INITIALIZE LSTM MODEL
    lstm = LSTM(model_name, model_dir, metrics_dir)

    path = model_dir + "/" + model_name

    if load_model == True:

        # search the model's folder
        is_model_dir = os.path.isdir(path)
        
        # if the model's folder already exists, load the model
        if is_model_dir == True:
            try:
                n_ts_in_loaded, n_features_loaded = lstm.load_model()
                print(f"LSTM model loaded from path: {path}")
            except:
                print(f"Error: Could not load LSTM model from path: {path}, exiting...")
                exit(1)
        
            if ((n_ts_in_loaded!=n_ts_in) or (n_features_loaded!=n_features)):
                n_ts_in = n_ts_in_loaded
                n_features = n_features_loaded
                print(f"Warning: LSTM parameters changed to fit loaded model")
        
        # if the model's folder does not exist, create a new model
        else:
            lstm.create_model(n_ts_in, n_features)
            print("Error: Model to load not found. New LSTM model created")
        
    else:
        lstm.create_model(n_ts_in, n_features)
        print("New LSTM model created")


    ##### ----- LOAD DATA ----- #####
    
    # x_train <-- data to be used for training the LSTM, e.g. stoppage specific type, number of rejects, etc.
    # y_train <-- labels (0 = no failures in the next N minutes, 1 = failure in the next N minutes) relative to x_train data
    # labels in y_train are actually one-hot encoded: label 0 becomes [1,0], label 1 becomes [0,1] since this is a binary
    # classification problem
    
    # The input to LSTM layer, i.e. x_train, should be of size (n_samples, n_ts_in, n_features).
    # N.B. x_train should contain real data, here we create an array of random numbers just to provide an example
    n_data_items = 3

    def createRandomData():
        upper_bound = 10
        return np.random.randint(upper_bound, size=(n_data_items, n_ts_in, n_features))

    x_train = createRandomData() 
    # the size of y_train should be (n_data_items, 2).
    # y_train should contain the real labels (one-hot encoded);
    # here we create an array of random labels just to provide an example
    def createRandomLabels():
        import random
        list_labels = []
        accepted_labels = [[0,1],[1,0]]
        for sample in range(n_data_items):
            list_labels.append(random.choice(accepted_labels))
        return np.array(list_labels)
    
    y_train = createRandomLabels()

    # TRAIN LSTM
    metrics_dict = lstm.train(x_train, y_train, batch_size, n_epochs)

    # SAVE MODEL AND METRICS
    lstm.save_model()
    lstm.save_metrics_all(metrics_dict)
    
    ##### ----- PREDICTION ----- #####
    
    # the size of a single data item used for prediction, i.e. x_pred, should be (1, n_ts_in, n_features)
    def createPredictionData():
        upper_bound = 10
        return np.random.randint(upper_bound, size=(1, n_ts_in, n_features))
    x_pred = createPredictionData()
    
    # use LSTM model to perform prediction on x_pred
    prediction = lstm.predict(x_pred)
    
    prediction_result = prediction[0]
    prediction_precision = prediction[1]
    print(f"Prediction: {prediction_result}, confidence: {prediction_precision}")