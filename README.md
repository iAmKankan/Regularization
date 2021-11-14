## Index

# Regularization
* Regularization is a technique that reduces overfitting, which occurs when neural networks attempt to memorize training data, rather than learn from it.
* Regularization is one of the techniques that can prevent overfitting. 
* A number of different regularization techniques exist. **Most work by analyzing and potentially modifying the weights of a neural network as it trains.**

* Types of Regularization
  * L1 (Lasso) Regularization
  * L2 (Ridge) Regularization
  * Dropout Regularization
  
  
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


