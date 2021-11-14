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


### Ridge Regression
![light](https://user-images.githubusercontent.com/12748752/141667908-4ec63aed-5cd0-4b35-9a45-3d52fba893b8.png)
* Ridge Regression (also called Tikhonov regularization) is a regularized version of Linear Regression: a regularization term equal to <img src="https://latex.codecogs.com/svg.image?\alpha\sum_{i=1}^{n}\&space;\theta^{2}_{i}" title="\alpha\sum_{i=1}^{n}\ \theta^{2}_{i}" /> is added to the cost function.
This forces the learning algorithm to not only fit the data but also keep the model
weights as small as possible. Note that the regularization term should only be added
to the cost function during training. Once the model is trained, you want to evaluate
the model’s performance using the unregularized performance measure.

![light](https://user-images.githubusercontent.com/12748752/141667908-4ec63aed-5cd0-4b35-9a45-3d52fba893b8.png)
