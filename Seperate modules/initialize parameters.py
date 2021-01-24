
    def initialize_params( X_input,layer_units[],layer_initializers[],W,b,Z,A):

       '''
        Initializes weight and bias matrices for training, using the desired initialization algorithm(s) 
        that will be chosen by the user whether to choose  to initialize weights by zeros or random intialization.

        :param X_input: Training set
               W:  weights of the layers
               b:  bias of the layers 
               Z:
               A:
               layer_initializers: array has the type of weights initiaizations of each layer
               layer_units: array has the number of neuron units in each layer

        
         
        '''
        

        layer_units[0] = X_input.shape[0]
        m = X_input.shape[-1]   

        num_layers = len(self.layer_units) - 1
        for l in range(1, num_layers+1):
            n_l = layer_units[l]
            n_prev = layer_units[l-1]

            if layer_initializers[l] == 'zeros':
                W.append(np.zeros(shape=(n_l, n_prev), dtype=np.float32))
                b.append(np.zeros(shape=(n_l, 1), dtype=np.float32))

            

            elif layer_initializers[l] == 'random':
                W.append(np.random.rand(n_l, n_prev).astype(np.float32))
                b.append(np.random.rand(n_l, 1).astype(np.float32))

           
            else:
                return False
            Z.append(np.zeros(shape=(n_l, m), dtype=np.float32)) 
            A.append(np.zeros(shape=(n_l, m), dtype=np.float32))
