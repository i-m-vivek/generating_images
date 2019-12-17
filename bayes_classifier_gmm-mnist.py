import cv2
import os
import glob
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.mixture import GaussianMixture

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


from sklearn.datasets import load_digits

data= load_digits()
images  = data.images
targets = data.target
images = images.reshape((-1, 64))
clf = BayesClassifier()
clf.fit(images, targets, num_components = 10)

for i in range(10):
    for j in range(10):
        sample, mean = clf.give_sample_given_class(clf.categories[i], get_mean= True)
        plt.imshow(sample[0].reshape(8, 8), cmap = "gray")
        plt.savefig(f"figtype-{str(i)}-{str(j)}")

