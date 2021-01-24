class Data():
    def shuffle_split_data(self, X, y, train_size=0.7):  # X: input data, y:labels, 70% of data for training
        import numpy as np
        arr_rand = np.random.rand(X.shape[0])  # shuffles data
        split = arr_rand < np.percentile(arr_rand, train_size * 100)  # takes 70% of data
        X_train = np.array(X[split]).T  # to reshape the data as required for model
        y_train = np.array(y[split]).ravel()  # ( m , ) instead of ( m, 1 )
        X_test = np.array(X[~split]).T  # takes the remaining 30% for testing
        y_test = np.array(y[~split]).ravel()
        return X_train, y_train, X_test, y_test

    def load_data(self, path, label):
        import pandas as pd
        data = pd.read_csv(path)  # read the csv data file
        Y = data[[label]]  # getting data labels
        data.drop([label], inplace=True, axis=1)  # dropping the label column
        X = data
        return self.shuffle_split_data(X, Y)