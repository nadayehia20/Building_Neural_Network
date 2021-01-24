def add_layer( units, activation='tanh', initializer='xavier',layer_units[],layer_activations[],layer_initializers[]):

        '''
        module that builds a layer knowing number of neurons in the layer,
        the activation function that will be used to it and the initializer type for weights/bias 
        :param units: number of neurons in this layer
        :param activation: Activation function for the neurons in this layer. 
                           Current options are: tanh, sigmoid, softmax, reLU, leaky reLU
           
        :param initializer: Algorithm to initialize weights/biases of the NN.
                            Current options are:
                           'xavier' : Sets weights/biases to numbers sampled from a normal distribution, with variance = 1/n_{l-1}
                           'zeros' : Sets all weights/biases to zero
                           'random' : Sets weights/biases to random numbers between 0 and 1, from a uniform distribution
                           
                           
        :param layer_activation: array has the activation function name that will be used for the layer
               layer_initializers: array has the type of weights initiaizations of each layer
               layer_units: array has the number of neuron units in each layer
        '''

        layer_units.append(units)
        layer_activations.append(activation)
        layer_initializers.append(initializer)
