## Index
![deep](https://user-images.githubusercontent.com/12748752/141667909-22520af3-61cf-4cbc-a8f5-f99947c9b10d.png)

## Regularization
![deep](https://user-images.githubusercontent.com/12748752/141667909-22520af3-61cf-4cbc-a8f5-f99947c9b10d.png)
* Deep neural networks typically can have several of thousands of parameters. 
* With so many parameters, the network has an incredible amount of freedom and can fit a huge variety of complex datasets. 
* But this great flexibility also means that it is prone to overfitting the training set.

* Regularization is a technique that reduces [**Overfitting.**](https://github.com/iAmKankan/MachineLearning_With_Python#overfitting-and-underfitting)

### Different regularization techniques
![light](https://user-images.githubusercontent.com/12748752/141667908-4ec63aed-5cd0-4b35-9a45-3d52fba893b8.png)

* One of the best regularization techniques is Early stopping. 
* Even though Batch Normalization was designed to solve the vanishing/exploding gradients problems, is also acts like a pretty good regularizer.
* Other popular regularization techniques for neural networks:
  * **ℓ1 and ℓ2 regularization ( L1 (Lasso) Regularization, L2 (Ridge) Regularization)**
  * **Dropout** 
  * **Max-norm regularization.**


## L1 Regularization (L1 = lasso):
 
 * The main objective of creating a model(training data) is making sure it fits the data properly and reduce the loss.
 * Sometimes the model that is trained which will fit the data but it may fail and give a poor performance during analyzing of data (test data). This leads to overfitting. Regularization came to overcome overfitting.
 
 
  * **Lasso Regression (Least Absolute Shrinkage and Selection Operator) adds “Absolute value of magnitude” of coefficient, as penalty term to the loss function.**
 * Lasso shrinks the less important feature’s coefficient to zero; thus, removing some feature altogether. 
 * So,this works well for feature selection in case we have a huge number of features.
 
 * Methods like Cross-validation, Stepwise Regression are there to handle overfitting and perform feature selection work well with a small set of features. 
 * These techniques are good when we are dealing with a large set of features.
* Along with shrinking coefficients, the lasso performs feature selection, as well. (Remember the ‘selection‘ in the lasso full-form?) Because some of the coefficients become exactly zero, which is equivalent to the particular feature being excluded from the model.


## L2 Regularization(L2 = Ridge Regression):

* **Ridge regression adds “squared magnitude of the coefficient" as penalty term to the loss function. Here the box part in the above image represents the L2 regularization element/term.**
* 


## R square(where to use and where not)
* R-squared is a statistical measure of how close the data are to the fitted regression line. It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression.

## Regularized Linear Models
* A good way to reduce overfitting is to regularize the model (i.e., to constrain it): the fewer degrees of freedom it has, the harder it will be for it to overfit the data. 
* For example, a simple way to regularize a polynomial model is to reduce the number of polynomial degrees.
* For a linear model, regularization is typically achieved by constraining the weights of the model. 
* We will now look at Ridge Regression, Lasso Regression, and Elastic Net,
which implement three different ways to constrain the weights.

