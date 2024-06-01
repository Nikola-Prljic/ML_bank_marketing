import tensorflow 
import keras
import numpy as np
import pandas as pd
from clean_data import get_input_layers
from logistic_regression import logisticRegression
import matplotlib.pyplot as plt


""" def train():
    keras.models.fit()

def model(X: np.ndarray, y: np.ndarray):
    #X = [[0., 0.], [1., 1.]]

    #y = [0, 1]
    train(X , y) """

    #clf.predict([[2., 2.]])
learning_rate=0.01
epochs=70
df = pd.read_parquet('preprocessed_bank_marketing.gzip')
features_names = ['age', 'balance', 'duration', 'job', 'loan', 'pdays']
model, epochs, hist = logisticRegression(df, features_names, learning_rate=learning_rate, epochs=epochs, batch_size=600, split=0.9)
plt.plot(epochs, hist['auc'], color='r', label='AUC')
plt.title('AUC OVER EPOCHS')
plt.legend()
plt.xlabel('EPOCHS')
plt.ylabel('AUC')
plt.grid(visible=True)
plt.show()

#df = pd.read_parquet('preprocessed_bank_marketing.gzip')
#X = np.array([df['duration'].to_numpy(dtype=np.float32), df['age'].to_numpy(dtype=np.float32)])
#X = df['duration'].to_numpy(dtype=np.float32)
#df['age'].to_numpy(dtype=np.float32)
#y = df['sub'].to_numpy(dtype=np.float32)
#sub_model(X, y)