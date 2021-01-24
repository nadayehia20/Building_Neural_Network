
def activation_func(X,activation):
    #implemented in activation_func.py
    return X



def forward_propagate(layer_units,W,b,Z,A,layer_activations):
    import numpy as np
    num_layers = len(layer_units) - 1  # get the number of layers except output layer
    for l in range(1, num_layers + 1):
        Z[l] = np.dot(W[l],A[l - 1]) + b[l]  # Z=WX+b
        A[l] = activation_func(Z[l], activation=layer_activations[l])  # A=Activation(Z)