## Index
![deep](https://user-images.githubusercontent.com/12748752/141667909-22520af3-61cf-4cbc-a8f5-f99947c9b10d.png)
* [Regularized Linear Models](#regularized-linear-models)
  * [Ridge Regression](#ridge-regression)
  * [Lasso Regression](#lasso-regression)
  * [Elastic Net](#elastic-net)
  * [When to choose which](#when-to-choose-which)
* [Early Stopping](#early-stopping)
* [References](#references)

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

#### Cost function
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
* The penalty hyperparameter sets the type of regularization term to use. 
* Specifying "l2" indicates that you want SGD to add a regularization term to the cost function equal to half the square of the ℓ2 norm of the weight vector: this is simply Ridge Regression.

### Lasso Regression
![light](https://user-images.githubusercontent.com/12748752/141667908-4ec63aed-5cd0-4b35-9a45-3d52fba893b8.png)
* Just like Ridge Regression, it adds a regularization term to the cost function, but it uses the ℓ1 norm of the weight vector instead of half the square of the ℓ2 norm.
 
#### Cost function
![light](https://user-images.githubusercontent.com/12748752/141667908-4ec63aed-5cd0-4b35-9a45-3d52fba893b8.png)
<img src="https://latex.codecogs.com/svg.image?J(\theta)=MSE(\theta)\&space;&plus;\&space;\alpha\sum_{i=1}^{n}\&space;|\theta_{i}|" title="J(\theta)=MSE(\theta)\ +\ \alpha\sum_{i=1}^{n}\ |\theta_{i}|" />

> #### An important characteristic of Lasso Regression is that it tends to completely eliminate the weights of the least important features (i.e., set them to zero).
> #### Lasso Regression automatically performs feature selection and outputs a sparse model (i.e., with few nonzero feature weights).

> ### Python Code
```python
 from sklearn.linear_model import Lasso
 lasso_reg = Lasso(alpha=0.1)
 lasso_reg.fit(X, y)
 lasso_reg.predict([[1.5]])
 
 [out] array([1.53788174])
```
### Elastic Net
![light](https://user-images.githubusercontent.com/12748752/141667908-4ec63aed-5cd0-4b35-9a45-3d52fba893b8.png)
* Elastic Net is a middle ground between Ridge Regression and Lasso Regression. 
* The regularization term is a **simple mix of both Ridge and Lasso’s regularization terms** and you can control the mix ratio _**r**_. 
> * When r = 0, Elastic Net is equivalent to Ridge Regression
> * When r = 1, it is equivalent to Lasso Regression

#### Cost Function
![light](https://user-images.githubusercontent.com/12748752/141667908-4ec63aed-5cd0-4b35-9a45-3d52fba893b8.png)

<img src="https://latex.codecogs.com/svg.image?J(\theta)=MSE(\theta)\&space;&plus;\&space;r\alpha\sum_{i=1}^{n}\&space;|\theta_{i}|&space;\&space;&plus;\&space;\frac{1-r}{2}\&space;\alpha\sum_{i=1}^{n}\&space;\theta_{i}^{2}" title="J(\theta)=MSE(\theta)\ +\ r\alpha\sum_{i=1}^{n}\ |\theta_{i}| \ +\ \frac{1-r}{2}\ \alpha\sum_{i=1}^{n}\ \theta_{i}^{2}" />

```python
 from sklearn.linear_model import ElasticNet
 elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
 elastic_net.fit(X, y)
 elastic_net.predict([[1.5]])
 [out] array([1.54333232])
 
```


### When to choose which
![light](https://user-images.githubusercontent.com/12748752/141667908-4ec63aed-5cd0-4b35-9a45-3d52fba893b8.png)
* It is almost always preferable to have at least a little bit of regularization, so generally you should avoid plain Linear Regression. 
* Ridge is a good default, but if you suspect that only a few features are actually useful, you should prefer Lasso or Elastic Net 
* Since they tend to reduce the useless features’ weights down to zero as we have discussed. 
* In general, Elastic Net is preferred over Lasso since Lasso may behave erratically when the **number of features is greater than the number of training instances** or **when several features are strongly correlated.**


### Early Stopping
![deep](https://user-images.githubusercontent.com/12748752/141667909-22520af3-61cf-4cbc-a8f5-f99947c9b10d.png)
> #### Early Stopping is a very different way to regularize iterative learning algorithms such as Gradient Descent is to stop training as soon as the validation error reaches a minimum.

* With **Stochastic** and **Mini-batch Gradient Descent**, the curves are not so smooth, and it may be hard to know whether you have reached the minimum or not. 
* One solution is to stop only after the validation error has been above the minimum for some time (when you are confident that the model will not do any better), then roll
back the model parameters to the point where the validation error was at a minimum.





### References
![deep](https://user-images.githubusercontent.com/12748752/141667909-22520af3-61cf-4cbc-a8f5-f99947c9b10d.png)

* [Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-Tensorflow- O’Reilly-Media-2019](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
