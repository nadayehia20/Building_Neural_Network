
    def initialize_params(self, X_input):

       '''
        Initializes weight and bias matrices for training, using the desired initialization algorithm(s) 
        that will be chosen by the user whether to choose  to initialize weights by zeros or random intialization.

        :param X_input: Training set
        '''
        import numpy as np

        self.layer_units[0] = X_input.shape[0]
        m = X_input.shape[-1]   

        num_layers = len(self.layer_units) - 1
        for l in range(1, num_layers+1):
            n_l = self.layer_units[l]
            n_prev = self.layer_units[l-1]

            if self.layer_initializers[l] == 'zeros':
                self.W.append(np.zeros(shape=(n_l, n_prev), dtype=np.float32))
                self.b.append(np.zeros(shape=(n_l, 1), dtype=np.float32))

            

            elif self.layer_initializers[l] == 'random':
                self.W.append(np.random.rand(n_l, n_prev).astype(np.float32))
                self.b.append(np.random.rand(n_l, 1).astype(np.float32))

           
            else:
                return False
            self.Z.append(np.zeros(shape=(n_l, m), dtype=np.float32)) 
            self.A.append(np.zeros(shape=(n_l, m), dtype=np.float32))
