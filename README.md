# Image Generation
This repository contains different methods for generating new images.

### Image Generation Using Multivariate Gaussian Distribution 
We can make our model to generate new images by learning a multivariate gaussian distribution, the parameters for the model that are mean and covariance will be same as the mean of the data for that class and covariance of the data. <br>
I trained my model on MNIST digit data. The code can be found in [bayes.py](https://github.com/i-m-vivek/generating_images/blob/master/bayes.py).


![Zero Mean](https://github.com/i-m-vivek/generating_images/blob/master/images/Bayes_Gen/zero.png "Zero Mean")
![Zero Generated](https://github.com/i-m-vivek/generating_images/blob/master/images/Bayes_Gen/zero_gen.png "Zero Generated")

Mean Image & Generated Image
