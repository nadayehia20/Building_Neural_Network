from NN_team21.Data import Data
from NN_team21.NeuralNetwork import NeuralNetwork
from NN_team21.utils import utils

nn = NeuralNetwork(loss='crossentropy', classifier='categorical', learning_rate=.02, metric='accuracy')
nn.add_layer(units=20, activation='tanh', initializer='xavier')
nn.add_layer(units=40, activation='tanh', initializer='xavier')
nn.add_layer(units=10, activation='softmax', initializer='xavier')
data_loader=Data()
path='train.csv'  # MNIST Data
label='label'
x_train, y_train, x_test, y_test=data_loader.load_data(path,label)
nn.train(x_train=x_train, y_train=y_train, iterations=150)
nn.visualize()
# Test NN on our test set
prediction = nn.test(x_test=x_test, y_test=y_test)
nn.evaluation_metric(prediction , y_test)

util=utils()
filename = 'final_model2.sav'
util.save_model(nn,filename)

x=util.load_model(filename)
pred=x.test(x_test,y_test)
x.evaluation_metric(pred,y_test)
