## Index
![deep](https://user-images.githubusercontent.com/12748752/141667909-22520af3-61cf-4cbc-a8f5-f99947c9b10d.png)
* [Regularization](#regularization)
* [Different regularization techniques](#different-regularization-techniques)
  * [ℓ1 and ℓ2 Regularization](#%E2%84%931-and-%E2%84%932-regularization)
  * [Dropout](#dropout)
  * [Monte-Carlo (MC) Dropout](#monte-carlo-mc-dropout)
  * [Max-Norm Regularization](#max-norm-regularization)
  * [Summary and Practical Guidelines](#summary-and-practical-guidelines)
  * [L1 Regularization (L1 = lasso)](#l1-regularization-l1--lasso)
  * [L2 Regularization(L2 = Ridge Regression)](#l2-regularizationl2--ridge-regression)
  * [R square(where to use and where not)](#r-squarewhere-to-use-and-where-not)
  * [Data Augmentation or Training Set Expansion](url)
* [Linear Regularization](https://github.com/iAmKankan/Regularization/blob/master/linear_regularization.md)
  * [Regularized Linear Models](https://github.com/iAmKankan/Regularization/blob/master/linear_regularization.md#regularized-linear-models)
  * [Ridge Regression](https://github.com/iAmKankan/Regularization/blob/master/linear_regularization.md#ridge-regression)
  * [Lasso Regression](https://github.com/iAmKankan/Regularization/blob/master/linear_regularization.md#lasso-regression)
  * [Elastic Net](https://github.com/iAmKankan/Regularization/blob/master/linear_regularization.md#elastic-net)
  * [When to choose which](https://github.com/iAmKankan/Regularization/blob/master/linear_regularization.md#when-to-choose-which)
* [Early Stopping](#early-stopping)

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
  * **ℓ1 and ℓ2 regularization ( [L1 (Lasso Regression), L2 (Ridge Regression)](https://github.com/iAmKankan/Regularization/blob/master/linear_regularization.md))**. Based on [**norms**](https://github.com/iAmKankan/Mathematics/blob/main/norm.md)
  * **Dropout** 
  * **Max-norm regularization.**

### ℓ1 and ℓ2 Regularization
![light](https://user-images.githubusercontent.com/12748752/141667908-4ec63aed-5cd0-4b35-9a45-3d52fba893b8.png)
* We can use ℓ1 and ℓ2 regularization to constrain a neural network’s connection weights (but typically not its biases).
> #### Apply ℓ2 regularization to a Keras layer’s connection weights, using a regularization factor of 0.01:
```python
layer = keras.layers.Dense(100, activation="elu", 
                           kernel_initializer="he_normal", 
                           kernel_regularizer=keras.regularizers.l2(0.01))
                         
```

* The l2() function returns a regularizer that will be called to compute the regularization loss, at each step during training. 
* This regularization loss is then added to the final loss. 

* You can just use **keras.regularizers.l1()** if you want ℓ1 regularization, and if you want both ℓ1 and ℓ2 regularization, use _**keras.regu
larizers.l1_l2()**_ (specifying both regularization factors).

```python
from functools import partial
RegularizedDense = partial(keras.layers.Dense,
                          activation="elu",
                          kernel_initializer="he_normal",
                          kernel_regularizer=keras.regularizers.l2(0.01))
model = keras.models.Sequential([
                     keras.layers.Flatten(input_shape=[28, 28]),
                     RegularizedDense(300),
                     RegularizedDense(100),
                     RegularizedDense(10, activation="softmax",
                     kernel_initializer="glorot_uniform")
```
 
> Since you will typically want to apply the same regularizer to all layers in your network,
as well as the same activation function and the same initialization strategy in all
hidden layers, you may find yourself repeating the same arguments over and over.
This makes it ugly and error-prone. To avoid this, you can try refactoring your code
to use loops. Another option is to use Python’s _**functools.partial()**_ function: it lets
you create a thin wrapper for any callable, with some default argument values. For










### Dropout
![light](https://user-images.githubusercontent.com/12748752/141667908-4ec63aed-5cd0-4b35-9a45-3d52fba893b8.png)


### Monte-Carlo (MC) Dropout
![light](https://user-images.githubusercontent.com/12748752/141667908-4ec63aed-5cd0-4b35-9a45-3d52fba893b8.png)


### Max-Norm Regularization
![light](https://user-images.githubusercontent.com/12748752/141667908-4ec63aed-5cd0-4b35-9a45-3d52fba893b8.png)


### Summary and Practical Guidelines
![deep](https://user-images.githubusercontent.com/12748752/141667909-22520af3-61cf-4cbc-a8f5-f99947c9b10d.png)



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
## Data Augmentation or Training Set Expansion

## References
