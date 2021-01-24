
class utils(): #for saving and loading model
  def load_model(filename):
    import pickle
    loaded_model = pickle.load(open(filename, 'rb')) #load the model with the last accuracy achieved
    return loaded_model

  def save_model(nn,filename):
    import pickle
    pickle.dump(nn, open(filename, 'wb')) #save the last accuracy achieved model