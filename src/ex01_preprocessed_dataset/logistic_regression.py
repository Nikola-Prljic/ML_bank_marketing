from clean_data import prepare_data
import tensorflow as tf

#from model.clean_data import prepare_data, get_input_layers
from keras import layers
import pandas as pd
import numpy as np
import datetime
import keras
from clean_data import get_input_layers
import matplotlib.pyplot as plt

def create_model(my_inputs: dict, my_learning_rate: int) -> keras.Model:
    
    # Combine layers
    concatenated_inputs = layers.Concatenate()(my_inputs.values())
    
    # Create hidden neural network layers
    x = layers.Dense(80, input_shape=(len(my_inputs),), activation='relu', name='32')(concatenated_inputs)
    x = layers.Dense(40, activation='relu', name='16')(x)
    x = layers.Dense(20, activation='relu', name='163')(x)
    x = layers.Dropout(rate=0.6, name='dropout')(x)

    # Create output layer because we have 4 possibilities
    output = layers.Dense(1, activation='sigmoid')(x)

    # Create Model and use a more accurate loss
    model = keras.Model(inputs=my_inputs, outputs=output)
    model.compile(optimizer=keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.AUC(num_thresholds=80, name='auc'),])
    
    #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    return model
    #, tensorboard_callback

def train_model(model: keras.Model, dataset: pd.DataFrame, label, epochs, batch_size=10, shuffle=True):
    
    # Create features dict
    features = {name:np.array(value, dtype=np.float16) for name, value in dataset.items()}
    
    # Fit everything into the model and set
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=shuffle)
                        #callbacks=[tensorboard_callback])
    
    
    # Save the loss and accuracy
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    return epochs, hist

def evaluate_model(model: keras.Model, train_df, test_df, test_label):

    # Evaluate create features dict
    test_features = {name:np.array(value) for name, value in test_df.items()}
    print('------------Evaluated Model------------')
    model.evaluate(test_features, test_label, batch_size=100, use_multiprocessing=True)

    # Predict and small test
    features = {name:np.array(value) for name, value in train_df.items()}
    predict_res = np.array(model.predict(features)).flatten()
    yes = [num for num in predict_res if num >= 0.6]
    no = [num for num in predict_res if num < 0.6]
    col = np.where(predict_res >= 0.6, 'r', 'g')
    #print(yes)
    plt.scatter(predict_res, np.arange(0, predict_res.size), color=col)
    #plt.hist(yes)
    #plt.hist(no)
    #cola = np.where(predict_res >= 0.6, 'N', 'Y')
    #plt.pie(cola, labels=['N', 'Y'], autopct='%1.1f%%', startangle=140)
    plt.show()
    #predict = np.argmax(predict[:1000], axis=1)
    #houses_dict = {0: 'no', 1: 'yes'}
    #predict = [houses_dict[house] for house in list(predict)]
    #print(predict_res[0:1000])

def logisticRegression(df, features_names, learning_rate, epochs, batch_size=10, split=0.8):
    
    # Prepare data, split Normalize and create labels
    train_df, test_df, train_label, test_label = prepare_data(df, split)
    #train_df, test_df =split_df(df, split=0.8)
    # Create a dict with the input labels
    inputs = get_input_layers(features_names)

    print(inputs)
    model = create_model(inputs, learning_rate)

    epochs, hist = train_model(model,
                               train_df, train_label, epochs, batch_size)

    # Eval :O
    evaluate_model(model, train_df, test_df, test_label)
    return model, epochs, hist
