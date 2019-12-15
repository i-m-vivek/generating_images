import numpy as np 
import cv2
import os
import glob
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt 

def load_devanagri(img_dir):
    """
    Imports the devanagri dataset 
    img_dir is the path where the data folder of different
    character are present 
    
    Returns: data, label 
    data: matrix containing data 
    label: corresponding label for each sample
    """
    dirs = os.listdir(img_dir)
    data = []
    label = []
    for i in range(len(dirs)):
        data_path = os.path.join(img_dir+ "\\"+dirs[i],'*g')
        files = glob.glob(data_path)
        for f1 in files:
            img = cv2.imread(f1)
            data.append(img)
            label.append(dirs[i])
    data = np.array(data)
    data = data[:, :, :, 2]
    data = np.reshape(data, (-1, 1024))
    label = np.array(label)
    return data, label



class BayesClassifier:
    def fit(self, X, y):
        self.categories = list(np.unique(y))
        self.params = []
        for i in range(len(self.categories)):
            X_cat = X[y==self.categories[i]]
            mean= np.mean(X_cat, axis = 0)
            cov = np.cov(X_cat.T)
            p = {"mean": mean, "cov": cov}
            self.params.append((self.categories[i],p))

    def give_sample_given_class(self, y):
        for i in range(len(self.categories)):
            if self.params[i][0] == y:
                param = self.params[i][1]
                break
            else:
                continue
            
        return mvn.rvs(mean = param["mean"], cov = param["cov"])
            
    def give_sample(self):
        choice = np.random.choice(self.categories)
        return self.give_sample_given_class(choice)
    
    def get_mean(self, y):
        for i in range(len(self.categories)):
            if self.params[i][0] == y:
                param = self.params[i][1]
                break
            else:
                continue
            
        return param["mean"]
        
        
img_dir = ""
X, Y = load_devanagri(img_dir)    
    
clf = BayesClassifier()
clf.fit(X, Y)

sample = clf.give_sample_given_class(clf.categories[-5])
plt.imshow(sample.reshape(32, 32), cmap = "gray")

mean = clf.get_mean(clf.categories[-5])
plt.imshow(mean.reshape(32, 32), cmap = "gray")