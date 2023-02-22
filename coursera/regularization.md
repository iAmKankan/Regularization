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

And so to add **regularization** to the **logistic regression**, we add **lambda** $\large{\color{Purple}\lambda}$ , which is called the $\large{\color{Purple}\textit{ Regularization Parameter}}$ . We add to the loss $\large{\color{Purple}\mathrm{\frac{\lambda}{2M}\parallel w \parallel_2^2}}$ . 

$\large{\color{Purple}\mathrm{Euclidean\ Norm\ \ or\ \ L^2\ norm\ :\ }}$ [link ↗️](https://github.com/iAmKankan/Mathematics/blob/main/LinearAlgebra/norms.md#1-euclidean-norm-or-pythagorean-norm-or-2-norm-or-l2)

$$\Huge{\color{Purple}\mathrm{\parallel w \parallel_2^2 = \Sigma^{n_x}_{i=1} w_j^2 = w^{\top}w}}$$

So here, the norm of **w squared** is just equal to **sum from j** equals 1 to **nx** of **wj squared**, or this can also be written **w transpose w**, it's just a square **Euclidean norm** of the prime to **vector w**. And this is called **L2 regularization**.

#### Why do you regularize just the parameter w? Why don't we add something here about b as well? $\large{\color{Purple}+ \frac{\lambda}{2m}b^2 }$ 
In practice, you could do this, but I usually just omit this. 
* Because if you look at your parameters $\large{\color{Purple}\textrm{w}}$ is usually a pretty **high dimensional** parameter **vector**, especially with a **high variance problem** . * Maybe w just has a lot of parameters, so you aren't fitting all the parameters well,
* Whereas $\large{\color{Purple}\textrm{b}}$ is just a **single number**. 
* So almost all the **parameters** are in $\large{\color{Purple}\textrm{w}}$  rather $\large{\color{Purple}\textrm{b}}$. And if you add this last term, in practice, it won't make much of a **difference**, because $\large{\color{Purple}\textrm{b}}$ is just one parameter over a very large number of parameters. 
So, I usually just don't bother to include it.

$\large{\color{Purple}\mathrm{Manhattan\ Norm\ \ or\ \ L^1\ norm\ :\ }}$ [link ↗️](https://github.com/iAmKankan/Mathematics/blob/main/LinearAlgebra/norms.md#2-manhattan-norm-or-1-norm-or--l1) If we use $L^1$ norm then

$$\Huge{\color{Purple}\Sigma^{n_x}_{i=1} |w|  =  \mathrm{\frac{\lambda}{2M}\parallel w \parallel_1}}$$


If you use **L1 regularization**, then w will end up being **sparse**. And what that means is that the **w vector** will have a **lot of zeros** in it. And some people say that this can help with **compressing the model**, because the **set of parameters are zero**, and you need **less memory** to store the model. Although, I find that, in practice, L1 regularization to make your model sparse, helps only a little bit. So I don't think it's used that much, at least not for the purpose of compressing your model. And when people train your networks, **L2 regularization** is just used much much more often.

 Lambda $\large{\color{Purple}\lambda$ here is called the **regularization parameter** and usually, you set this using your **development set**, or using **dev set** or **cross validation**. When you a variety of values and see what does the best, in terms of trading off between doing well in your training set versus also setting that two normal of your parameters to be small. Which helps prevent **over fitting**. So **lambda** is **another hyper parameter** that you might have to tune. 

$\large{\color{Purple}\textrm{Frobenius norm (Matrix norm)}}$ [link ↗️](https://github.com/iAmKankan/Mathematics/blob/main/LinearAlgebra/norms.md#frobenius-norm)