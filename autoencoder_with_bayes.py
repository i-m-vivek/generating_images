from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.mixture import GaussianMixture

encoding_dim = 32
num_samples = 1000
BATCH_SIZE =  128
num_epochs = 50
num_components = 10

input_img = Input(shape = (784, ))
encoded = Dense(encoding_dim, activation = 'relu')(input_img)
decoded = Dense(784, activation = "sigmoid")(encoded)
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape = (encoding_dim, ))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer = "adam", loss = "binary_crossentropy")


(X_train, y_train),(X_test, y_test) = mnist.load_data()

X_train = X_train.astype("float32")/255
X_test = X_test.astype("float32")/255

X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

# print(X_train.shape)
# print(X_test.shape)

autoencoder.fit(X_train, X_train,
                epochs = num_epochs, 
                batch_size = BATCH_SIZE,
                shuffle = True,
                validation_data = (X_test, X_test))


class BayesClassifier:
  def fit(self, X, y, num_components):
    self.categories = list(np.unique(y))
    self.params = []

    for i in range(len(self.categories)):
      X_cat = X[y == self.categories[i]]
      model = GaussianMixture(n_components=num_components)
      model.fit(X_cat)
      mean = np.mean(X_cat, axis = 0)
      self.params.append((self.categories[i], model, mean))

  def give_sample_given_class(self, y, get_mean = False):
    for i in range(len(self.categories)):
      if self.params[i][0] == y:
        if get_mean == True:
          return self.params[i][1].sample(), self.params[i][2]
        else: 
          return self.params[i][1].sample()

clf = BayesClassifier()
clf.fit(X_train[:num_samples], y_train[:num_samples], num_components = num_components)

n = 10
plt.figure(figsize = (20, 4))

for i in range(n):
  ax = plt.subplot(2, n, i+1)
  sample= clf.give_sample_given_class(clf.categories[i])
  plt.imshow(sample[0].reshape(28, 28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  encoded_img = encoder.predict(sample[0])
  decoded_img = decoder.predict(encoded_img)

  ax = plt.subplot(2, n, i+1+n)
  plt.imshow(decoded_img.reshape(28, 28), cmap = "gray")
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

plt.show()