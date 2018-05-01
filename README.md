# Active Learning
### https://en.wikipedia.org/wiki/Active_learning_(machine_learning)

The goal of this project is to provide a framework for users to build classifiers trained on datasets that are expensive or time-consuming to fully label (ex: a scenario where researchers must hand-label each instance) by choosing the most informative instances to train on. The steps we plan on taking are the following:
  1. The model will first be trained on a small subsample of the dataset (that has been labeled) which includes an equal number of examples from each class. 
  2. The model will then give its predictions for the entirety of the unlabeled dataset. 
  3. The model will then determine a subset of the testing data to be labeled by the user. This determination can be made in a number of ways:
      - The points which the model is most unsure about
      - The points which would change the current model the most
      - If using a committee of models (multiple models), the points the committee disagrees on most
  4. The model will then be trained on the original labeled data and the newly labeled data.
  5. Repeat steps 2-4 until accuracy is high enough or no more effort can be applied towards hand-labelling instances.
  6. A subsample of labeled data will be held out from the above process and used for evaluating “test” accuracy.

This project has been created using Python 3.6.
The dataset we have chosen is the MNIST digit classification dataset: https://www.kaggle.com/c/digit-recognizer/data

![alt text]Active_Learning/src/graphing/graph_outputs/Screenshot 2018-05-01 11.52.17.png
