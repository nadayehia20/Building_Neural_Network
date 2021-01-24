

def calculate_loss(actual,A,classifier,loss_function,loss,loss_history):
    import numpy as np
    m = actual.shape[-1]  # size of the label
    predicted = A[-1]  # output of the last layer

    if loss_function == 'crossentropy' and classifier == 'binary':
        loss_vector = -actual * np.log(predicted) - (1 - actual) * np.log(
            1 - predicted)  # loss=-(ylog(y_hat)+(1-y)log(1-y_hat))
    elif loss_function == 'crossentropy' and classifier == 'categorical':
        loss_vector = np.sum(-actual * np.log(predicted), axis=0, keepdims=True)  # loss=-ylog(y_hat)

    loss = (1 / m) * np.sum(loss_vector)  # total loss=(1/m)*sum for the loss vector
    loss_history.append(loss)  # loss_history stores all losses
