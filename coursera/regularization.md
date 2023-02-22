## Index
![light](https://user-images.githubusercontent.com/12748752/217685784-cbb25d67-8d84-47c2-89d1-b9737433fa4d.png)
![deep](https://user-images.githubusercontent.com/12748752/217685787-3c74eedd-9626-42ff-bab4-5e6ff825a9a4.png)


<!---
$\large{\color{Purple}\textit{l'heure}}$


<p align="center">
  <img src="" width=70%/>
  <br>
  <ins><b>     xxxxxx  </b></ins>
</p>
--->
## Regularization
![deep](https://user-images.githubusercontent.com/12748752/217685787-3c74eedd-9626-42ff-bab4-5e6ff825a9a4.png)
If you suspect your **neural network** is **overfitting** your data. That is you have a **high variance** problem, one of the first things you should try per probably
1.  **Regularization**. 
2. The other way to get **more training data** that's also quite reliable. 

But you can't always get more **training data**, or it could be **expensive to get more data**. But adding **regularization** will often help to prevent **overfitting**, or to reduce the errors in your network. So let's see how **regularization** works.

$\Large{\color{Purple}\textrm{Logistic Regression: }}$
Let's develop these ideas using logistic regression.For logistic regression, you try to **minimize** the cost function $\large{\color{Purple}\textrm{J}}$, which is defined as this **cost function**. Some of your training examples of the **losses** of the individual predictions in the different examples, where you recall that **w** and **b** in the logistic regression, are the parameters. 
* $\large{\color{Purple}\textrm{w}}$ is $\large{\color{Purple}n_x}$ dimensional **parameter vector**, $\Large{\color{Purple}\mathrm{w \in \mathbb{R}^{n_x}}}$
* $\large{\color{Purple}\textrm{b}}$ is a **real number**. $\Large{\color{Purple}\mathrm{b \in \mathbb{R}}}$

$$ \Huge{\color{Purple}\mathrm{J(w,b) = \frac{1}{M} \Sigma^M_{i=1} L (\hat{y}^{(i)},y^{(i)})}} {\color{Cyan} + \mathrm{\frac{\lambda}{2M}\parallel w \parallel_2^2}}$$

And so to add regularization to the logistic regression, what you do is add to it this thing, lambda, which is called the regularization parameter. I'll say more about that in a second. But lambda/2m times the norm of w squared. So here, the norm of w squared is just equal to sum from j equals 1 to nx of wj squared, or this can also be written w transpose w, it's just a square Euclidean norm of the prime to vector w. And this is called L2 regularization.
