## Index
![deep](https://user-images.githubusercontent.com/12748752/141667909-22520af3-61cf-4cbc-a8f5-f99947c9b10d.png)

## Regularized Linear Models
![deep](https://user-images.githubusercontent.com/12748752/141667909-22520af3-61cf-4cbc-a8f5-f99947c9b10d.png)

* A good way to reduce overfitting is to regularize the model (i.e., to constrain it): the fewer degrees of freedom it has, the harder it will be for it to overfit the data. 
* **For example**, a simple way to regularize a polynomial model is to reduce the number of polynomial degrees.
* For a linear model, regularization is typically achieved by **constraining the weights** of the model. 
> #### Regularized Linear Models to implement three different ways to constrain the weights:
> * **Ridge Regression**
> * **Lasso Regression**
> * **Elastic Net**

> #### It is important to scale the data (e.g., using a StandardScaler) before performing **Ridge Regression**, **Lasso Regression** as it is sensitive to the scale of the input features. This is true of most regularized models.

### Ridge Regression
![light](https://user-images.githubusercontent.com/12748752/141667908-4ec63aed-5cd0-4b35-9a45-3d52fba893b8.png)
* Ridge Regression (also called Tikhonov regularization) is a regularized version of Linear Regression: 
* A regularization term equal to <img src="https://latex.codecogs.com/svg.image?\alpha\sum_{i=1}^{n}\&space;\theta^{2}_{i}" title="\alpha\sum_{i=1}^{n}\ \theta^{2}_{i}" /> is added to the cost function.
* This forces the learning algorithm to not only fit the data but also keep the model weights as small as possible.
* Note that the regularization term should only be added to the cost function during training. 
* Once the model is trained, you want to evaluate the model’s performance using the unregularized performance measure.

#### Ridge Regression cost function
![light](https://user-images.githubusercontent.com/12748752/141667908-4ec63aed-5cd0-4b35-9a45-3d52fba893b8.png)
<img src="https://latex.codecogs.com/svg.image?J(\theta)=MSE(\theta)\&space;&plus;\&space;\alpha\sum_{i=1}^{n}\&space;\theta^{2}_{i}" title="J(\theta)=MSE(\theta)\ +\ \alpha\sum_{i=1}^{n}\ \theta^{2}_{i}" />

> * The hyperparameter α controls how much you want to regularize the model. 
> * If α = 0 then Ridge Regression is just Linear Regression. 
> * If α is very large, then all weights end up very close to zero and the result is a flat line going through the data’s mean.
> * Note that the bias term <img src="https://latex.codecogs.com/svg.image?\theta_0" title="\theta_0" /> is not regularized (the sum starts at i = 1, not 0).

> #### Python Code
```python
 from sklearn.linear_model import Ridge
 ridge_reg = Ridge(alpha=1, solver="cholesky")
 ridge_reg.fit(X, y)
 ridge_reg.predict([[1.5]])
 
 [out] array([[1.55071465]])
 ```
> ### Using Stochastic Gradient Descent
```python
 sgd_reg = SGDRegressor(penalty="l2")
 sgd_reg.fit(X, y.ravel())
 sgd_reg.predict([[1.5]])
 
 [out] array([1.47012588])

```

### References
![deep](https://user-images.githubusercontent.com/12748752/141667909-22520af3-61cf-4cbc-a8f5-f99947c9b10d.png)

* [Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-Tensorflow- O’Reilly-Media-2019](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
