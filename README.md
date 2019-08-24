# Hyperparameters: 
Model optimization is one of the toughest challenges in the implementation of machine learning solutions. Entire branches of machine learning and deep learning theory have been dedicated to the optimization of models. Typically, we think about model optimization as a process of regularly modifying the code of the model in order to minimize the testing error. However, deep learning optimization often entails fine tuning elements that live outside the model but that can heavily influence its behavior. Deep learning often refers to those hidden elements as hyperparameters as they are one of the most critical components of any machine learning application.   
Hyperparameters are - **** 1) Learning Rate, 2)  Number of Hidden Units, 3)Convolution Kernel Width ****

* Applying Hyperparametrs correctly is a Itaretive process.
# Training The model:
Data ->|1)Train set |2) Dev/holdon data/cross validation/Development set|3) Test
--|---|---|--

In retro style Neural Network we used Data in   

Train set | Dev | Test
---|---|---
70%|30|

Or

Train set | Dev | Test
---|---|---
60%|20|20

# Bias and Varience:

![Bias](/images/bias1.png)

### High Bias:
If we are using something linear algorithms to fit to a data like logestic regrassion, which is not very good fit; That makes High Bias or Underfitting the data.
### High Varience:
By using any complex classifier like Deep Neural Network or like so we fit the data perfectly thats makes Overfitting the data.


## Solution:
High Bias:(training data performence) |     1) Bigger Network 2) Train Longer 3) NN archetecture search
---|---

High Verience:(Dev set Performence) | 1) More data 2) Regularization 3) NN archetecture search
---|---

To achieve the **Low Bias** and **Low Varience**
## Why is Bias Variance Tradeoff?
The goal of any supervised machine learning algorithm is to achieve low bias and low variance. In turn the algorithm should achieve good prediction performance.

You can see a general trend in the examples above:

* Parametric or linear machine learning algorithms often have a high bias but a low variance.
* Non-parametric or non-linear machine learning algorithms often have a low bias but a high variance.

The parameterization of machine learning algorithms is often a battle to balance out bias and variance.

Below are two examples of configuring the bias-variance trade-off for specific algorithms:

* The k-nearest neighbors algorithm has low bias and high variance, but the trade-off can be changed by increasing the value of k which increases the number of neighbors that contribute t the prediction and in turn increases the bias of the model.
* The support vector machine algorithm has low bias and high variance, but the trade-off can be changed by increasing the C parameter that influences the number of violations of the margin allowed in the training data which increases the bias but decreases the variance.

There is no escaping the relationship between bias and variance in machine learning.

* **Increasing the bias will decrease the variance.**
* **Increasing the variance will decrease the bias.**

There is a trade-off at play between these two concerns and the algorithms you choose and the way you choose to configure them are finding different balances in this trade-off for your problem.

In reality, we cannot calculate the real bias and variance error terms because we do not know the actual underlying target function. Nevertheless, as a framework, bias and variance provide the tools to understand the behavior of machine learning algorithms in the pursuit of predictive performance.
