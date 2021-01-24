    




    def train(loss,A,classifier, x_train, y_train, iterations=1):
        import numpy as np
        #X_train: nXm array, n=number of features, m=training examples
        #y_train 1D array of size m
        if classifier == 'categorical':
            if y_train.ndim == 1: #makes sure it's a 1D array
                one_hot_array = np.zeros(shape=(y_train.max()+1, y_train.size)) #first step in transforming the one y data into a one hot encoded y data of size=cetrgoriesXm, initialize the matrix to zero
                one_hot_array[y_train, np.arange(y_train.size)] = 1 #puts 1 in the corresponding column of category
                y_train = one_hot_array
        if classifier == 'binary':
            if y_train.ndim == 1: 
                y_train = np.expand_dims(y_train, axis=0) #expands the shape of y_train


        A[0] = x_train
        initialize_params(x_train)

        print('Training neural network.')
        for i in range(iterations):

            forward_propagate()
            calculate_loss(y_train)
            sbackward_propagate(y_train)

            print_step_size = int(iterations/100) if iterations > 100 else 1
            if i % print_step_size == 0:
                print('At iteration {}, J = {}'.format(i, loss))

    def test(A,classifier, x_test, y_test):
        import numpy as np
        if y_test.ndim == 1:
            y_test = np.expand_dims(y_test, axis=0)

        if classifier == 'categorical':
            if y_test.ndim == 1:
                one_hot_array = np.zeros(shape=(y_test.max()+1, y_test.size))
                one_hot_array[y_test, np.arange(y_test.size)] = 1
                y_test = one_hot_array
        if classifier == 'binary':
            if y_test.ndim == 1:
                y_test = np.expand_dims(y_test, axis=0)

        A[0] = x_test
        forward_propagate()
        return A[-1] #returns Y, last value in A