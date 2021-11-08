from tensorflow import keras
import numpy as np
import json
from pathlib import Path
from typing import Tuple

class LSTM(object):

    def __init__(self, name:str, model_dir:str, metrics_dir:str):
        """ Constructor. Initialize the model to None and set the model's name.
        """
        self.model = None
        self.name = name
        self.model_dir = model_dir
        self.metrics_dir = metrics_dir

    def create_model(self, n_ts_in:int, n_features:int):
        """ Create a new model from scratch.
        """

        # the input to LSTM layer should be (n_samples, n_ts_in, n_features)

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.BatchNormalization(input_shape=(n_ts_in, n_features),
                                                    scale=False,
                                                    center=False)) # layer to normalize input samples
        self.model.add(keras.layers.LSTM(64, # dimensionality of the output space (number of neurons)
                                        activation="tanh",
                                        return_sequences=True)) # the output becomes the input of the next layer
        self.model.add(keras.layers.Dropout(0.5)) # apply dropout between layers to prevent overfitting
        self.model.add(keras.layers.LSTM(32, activation="tanh"))
        self.model.add(keras.layers.Dropout(0.5))
        #self.model.add(keras.layers.LSTM(16, activation="tanh"))
        #self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(10, activation="elu"))
        self.model.add(keras.layers.Dense(2, activation="softmax"))
        # softmax activation in the last layer and categorical crossentropy allow to get a result
        # telling something about the likelihood that a data point belongs to each of two classes.

        # configure the model for training
        self.model.compile(loss="categorical_crossentropy",
                        optimizer=keras.optimizers.Adam(lr=0.01),
                        metrics=["accuracy"])

        # self.model.summary()

        # save model
        self.save_model()

        # create folder for the metrics, if it does not exist
        path = self.metrics_dir + "/" + self.name
        Path(path).mkdir(parents=True, exist_ok=True)

        # create csv file to save metrics values
        with open(path + "/" + self.name + "_metrics.csv", "w") as output_file:
            output_file.write("epoch,loss,accuracy\n")

        return

    def save_model(self):
        """ Save current model's architecture and weights.
        """
        path = self.model_dir + "/" + self.name

        # create folder for the model, if it does not exist
        Path(path).mkdir(parents=True, exist_ok=True)

        # save model's architecture
        model_json_dict = json.loads(self.model.to_json())

        with open(path + "/" + self.name + ".json", "w") as output_file:
            json.dump(model_json_dict, output_file)

        # save model's weights
        self.save_model_weights()

        return

    def save_model_weights(self):
        """ Save current model's weights.
        """
        path = self.model_dir + "/" + self.name
        self.model.save_weights(path + "/" + self.name + ".hdf5")

        return

    def load_model(self) -> Tuple[int,int]:
        """ Load an already existing model.
            Return parameters (n_ts_in, n_features), which are retrieved from the loaded model.
        """

        path = self.model_dir + "/" + self.name
        n_ts_in = 0
        n_features = 0

        # load model's architecture
        with open(path + "/" + self.name + ".json", "r") as input_file:
            model_json_dict = json.load(input_file)
        
        for layer in model_json_dict["config"]["layers"]:
            if layer["class_name"] == "InputLayer":
                batch_input_shape = layer["config"]["batch_input_shape"]
                n_ts_in = batch_input_shape[1]
                n_features = batch_input_shape[2]

        self.model = keras.models.model_from_json(json.dumps(model_json_dict))

        # load model's weights
        self.model.load_weights(path + "/" + self.name + ".hdf5")

        self.model.compile(loss="categorical_crossentropy",
                optimizer=keras.optimizers.Adam(lr=0.01),
                metrics=["accuracy"])

        # self.model.summary()

        return n_ts_in, n_features
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, batch_size:int, n_epochs:int) -> dict:
        """ Train the model on the data that is passed as parameter.
            Return a dictionary with the values of the metrics.
        """

        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, validation_split=0.1)
        loss_list = history.history["loss"]
        accuracy_list = history.history["accuracy"]
        metrics_dict = {}
        metrics_dict["loss"] = loss_list
        metrics_dict["accuracy"] = accuracy_list

        return metrics_dict

    def predict(self, x: np.ndarray) -> Tuple[int,float]:
        """ Generate predictions for one single input sample.
            Return result and precision.
        """
        prediction = self.model.predict(x, batch_size=1)

        probability_0 = prediction[0,0]
        probability_1 = prediction[0,1]

        if probability_0 >= probability_1:
            prediction_result = int(0)
            prediction_precision = probability_0
        else:
            prediction_result = int(1)
            prediction_precision = probability_1

        return prediction_result, prediction_precision
    
    def save_metrics_all(self, metrics_dict: dict):
        """ Save metrics to CSV file with the corresponding metrics history.
        """
        path = self.metrics_dir + "/" + self.name

        with open(path + "/" + self.name + "_metrics.csv", "a+") as output_file:
            list_loss = metrics_dict["loss"]
            list_accuracy = metrics_dict["accuracy"]
            for epoch in range(1,len(list_loss)+1):
                metrics_string = str(epoch) + "," + str(list_loss[epoch-1]) + "," + str(list_accuracy[epoch-1]) + "\n"
                output_file.write(metrics_string)

        return






