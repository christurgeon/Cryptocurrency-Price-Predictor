import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

DEVDATA_PERCENT = 0.10 # Percent of sequential data we want saved for dev set

N_PERIODS_IN  = 30 # Number of periods looking backwards for training
N_PERIODS_OUT = 5  # Number of periods ahead to predict
N_FEATURES    = 1  # Number of features to train (price)


def scale(df):
    # Use this function to scale data to [0,1] range
    # y = (x - min) / (max - min)
    pass


def preprocess(df):
    pass


if __name__ == "__main__":

    dataframe = pd.DataFrame()
    for subdir, dirs, files in os.walk(r"crypto_data"):
        for file in files:
            filename = subdir + os.sep + file
            df = pd.read_csv(filename, names=["time", "low", "high", "open", "close", "volume"])
            df.set_index("time", inplace=True)
            
            print(filename, df.head(), sep='\n')

            # Retrieve the last 10% of data for dev set 
            times = sorted(df.index.values)
            split_index = times[-int(DEVDATA_PERCENT * len(times))]
            train_df = df[(df.index < split_index)]
            dev_df = df[(df.index >= split_index)]

            print("Train Size: {}Dev Size: {}\n".format(train_df.size, dev_df.size))

            # train_x, train_y = preprocess(train_df)
            # dev_x, dev_y = preprocess(dev_df)


