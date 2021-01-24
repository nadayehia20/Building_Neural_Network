    def visualize():
      import matplotlib.pyplot as plt
      plt.plot(loss_history)
      plt.title('Loss Vs iterations')
      plt.xlabel('iterations')
      plt.ylabel('L')
      plt.show()
