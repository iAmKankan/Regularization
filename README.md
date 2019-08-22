# Hyperparameter-tuning-Regularization
* Applying Hyperparametrs correctly is a Itaretive process.

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

![Bias](/images/1.png)

### High Bias:
If we are using something linear algorithms to fit to a data like logestic regrassion, which is not very good fit; That makes High Bias or Underfitting the data.
### High Varience:
By using any complex classifier like Deep Neural Network or like so we fit the data perfectly thats makes Overfitting the data.


## Solution:
High Bias: |     1)Bigger Network  
2)Train Longer    
3) NN archetecture search
---|---
