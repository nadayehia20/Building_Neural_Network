


    def backward_propagate_last_layer(A,loss_function, classifier,actual):
        import numpy as np
        if classifier=='categorical':
            if loss_function=='crossentropy':
                return -actual*(1 - A[-1])
        elif classifier=='binary':
            if loss_function=='crossentropy':
                return -actual + A[-1]



    def backward_propagate(actual,layer_units,activation_func,layer_activations,W,Z,A,b,learning_rate):
        import numpy as np
        m = actual.shape[-1]
        num_layers = len(layer_units) - 1

        # Iterate from layer L-1 to layer 1
        for l in range(num_layers, 0, -1):
            if l == num_layers:
                dZ = backward_propagate_last_layer(actual)
            else:
                g = activation_func(Z[l], activation=layer_activations[l], return_derivative=True)
                dZ = g*np.dot(np.transpose(W[l+1]), dZ) 

            dW = (1/m)*np.dot(dZ, np.transpose(A[l-1]))
            dB = (1/m)*np.sum(dZ, axis=1, keepdims=True)

            # Update parameters
            W[l] = W[l] - learning_rate*dW
            b[l] = b[l] - learning_rate*dB
            
            
            
            

