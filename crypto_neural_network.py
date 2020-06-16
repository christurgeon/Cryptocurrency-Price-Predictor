import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from collections import deque

MODEL_NAME = f""
USE_MINMAX_PREPROCESSING = False
DEVDATA_PERCENT = 0.15 # Percent of sequential data we want saved for dev set

N_PERIODS_IN  = 45 # Number of periods looking backwards for training (minutes)
N_PERIODS_OUT = 5  # Number of periods ahead to predict (minutes)
N_FEATURES    = 1  # Number of features to train (price)
EPOCHS        = 10 # Number of epochs to train model for
BATCH_SIZE    = 64 # Size of the training batch

def preprocess_minmax(df):
    df = df.drop("future", axis=1)
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    ret_na = df_scaled.dropna(inplace=True)
    print(df_scaled.head())
    print("Dropped NAs:\n{}".format(ret_na))
   
    return 1, 2



def preprocess_percent_change(df):
    # This function is based on Youtuber `sentdex` 
    # video on creating a cryptocurrency prediction 
    # mechanism using a Recurrent Neural Network.
    df = df.drop("future", axis=1)

    # Get the percent change and sale, drop the NA values
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)

    # 
    sequential_data = list()
    previous_days = deque(maxlen=N_PERIODS_IN)
    for i in df.values:
        previous_days.append([j for j in i[:-1]])
        if len(previous_days) == N_PERIODS_IN:
            sequential_data.append([np.array(previous_days), i[-1]])
    random.shuffle(sequential_data)

    # Need to balance buying and selling
    buys = sells = []
    for seq, trg in sequential_data:
        buys.append([seq, trg]) if trg else sells.append([seq, trg])
    lower = min(len(buys), len(sells))
    buys  = buys[:lower]
    sells = sells[:lower]
    sequential_data = buys + sells
    random.shuffle(sequential_data)

    # Separate X and Y 
    X = Y = []
    for seq, trg in sequential_data:
        X.append(seq)
        Y.append(trg)
    return np.array(X), np.array(Y)


def model():
    pass


if __name__ == "__main__":

    dataframe = pd.DataFrame()
    for subdir, dirs, files in os.walk(r"crypto_data"):
        for file in files:
            filename = subdir + os.sep + file
            df = pd.read_csv(filename, names=["time", "low", "high", "open", "close", "volume"])
            df.set_index("time", inplace=True)
            df["future"] = df["close"].shift(-N_PERIODS_OUT)
            df["target"] = df.apply(lambda x : int(x.future > x.close), axis=1)
            
            print(filename, df.head(15), sep='\n')

            # Split into training and dev data sets
            times = sorted(df.index.values)
            split_index = times[-int(DEVDATA_PERCENT * len(times))]
            train_df = df[(df.index < split_index)]
            dev_df = df[(df.index >= split_index)]

            print("Train Size: {}\nDev Size: {}\n".format(train_df.size, dev_df.size))

            if USE_MINMAX_PREPROCESSING:
                train_x, train_y = preprocess_minmax(train_df)
                dev_x, dev_y = preprocess_minmax(dev_df)
            else:
                train_x, train_y = preprocess_percent_change(train_df)
                dev_x, dev_y = preprocess_percent_change(dev_df)
                unique, counts = np.unique(train_y, return_counts=True)
                temp = dict(zip(unique, counts))
                print("Not buy: {}\nBuys: {}".format(temp[0], temp[1]))
                unique, counts = np.unique(dev_y, return_counts=True)
                temp = dict(zip(unique, counts))
                print("DEV Not buy: {}\t\tDEV buys:".format(temp[0], temp[1]))
            break


