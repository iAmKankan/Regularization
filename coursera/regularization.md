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
2. The other way to get **more training data** that's also quite **reliable**. Sometimes not possible to get more **training data** or it could be **very expensive**.

> ### “ Regularization is any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error.”
> ${\color{Purple}\textit{-Ian Goodfellow}}$

$\Large \textit{In other words: }$ **regularization** can be used to train models that **generalize** better on unseen data, by preventing the algorithm from overfitting the training dataset.


### ⚛️ $\Large{\color{Purple}\underline{\textbf{Logistic Regression:}}\ }$ 
### So how can we modify the _logistic regression_ algorithm to reduce the generalization error?
$\large Answer:$ It can be proven that **L2** or **Gauss** or **L1** or **Laplace** regularization have an equivalent impact on the algorithm. There are two approaches to attain the regularization effect.
1. **First approach:** adding a Regularization Term.
2. **Second approach:** Bayesian view of Regularization.



$\Large{\color{Black}\underline{\textbf{1. Adding a Regularization Term:}}\ }$ 

Let's develop these ideas using logistic regression.For logistic regression, you try to **minimize** the cost function $\large{\color{Purple}\textrm{J}}$, which is defined as this **cost function**. Some of your training examples of the **losses** of the individual predictions in the different examples, where you recall that **w** and **b** in the logistic regression, are the parameters. 
* $\large{\color{Purple}\textrm{w}}$ is $\large{\color{Purple}n_x}$ dimensional **parameter vector**, $\Large{\color{Purple}\mathrm{w \in \mathbb{R}^{n_x}}}$
* $\large{\color{Purple}\textrm{b}}$ is a **real number**. $\Large{\color{Purple}\mathrm{b \in \mathbb{R}}}$

$$ \Huge{\color{Purple}\mathrm{J(w,b) = \frac{1}{M} \Sigma^M_{i=1} L (\hat{y}^{(i)},y^{(i)})}} {\color{Cyan} + \mathrm{\frac{\lambda}{2M}\parallel w \parallel_2^2}}$$

And so to add **regularization** to the **logistic regression**, we add **lambda** $\large{\color{Purple}\lambda}$ , which is called the $\large{\color{Purple}\textit{ Regularization Parameter}}$ .
* We add to the loss $\large{\color{Purple}\mathrm{\frac{\lambda}{2M}\parallel w \parallel_2^2}}$ . Which is a **Euclidean norm** or $\mathbf{L^2}$ **norm**

$\large{\color{Purple}\mathrm{Euclidean\ Norm\ \ or\ \ L^2\ norm\ :\ }}$ [link ↗️](https://github.com/iAmKankan/Mathematics/blob/main/LinearAlgebra/norms.md#1-euclidean-norm-or-pythagorean-norm-or-2-norm-or-l2)

$$\Huge{\color{Purple}\mathrm{\parallel w \parallel_2^2 = \Sigma^{n_x}_{i=1} w_j^2 = w^{\top}w}}$$

So here, the norm of **w squared** is just equal to **sum from j** equals 1 to **nx** of **wj squared**, or this can also be written **w transpose w**, it's just a square **Euclidean norm** of the prime to **vector w**. And this is called **L2 regularization**.

#### Why do you regularize just the parameter w? Why don't we add something here about b as well? $\large{\color{Cyan}+ \frac{\lambda}{2m}b^2 }$ 
In practice, you could do this, but I usually just omit this. 
* Because if you look at your parameters $\large{\color{Purple}\textrm{w}}$ is usually a pretty **high dimensional** parameter **vector**, especially with a **high variance problem** . * Maybe w just has a lot of parameters, so you aren't fitting all the parameters well,
* Whereas $\large{\color{Purple}\textrm{b}}$ is just a **single number**. 
* So almost all the **parameters** are in $\large{\color{Purple}\textrm{w}}$  rather $\large{\color{Purple}\textrm{b}}$. And if you add this last term, in practice, it won't make much of a **difference**, because $\large{\color{Purple}\textrm{b}}$ is just one parameter over a very large number of parameters. 
So, I usually just don't bother to include it.

$\large{\color{Purple}\mathrm{Manhattan\ Norm\ \ or\ \ L^1\ norm\ :\ }}$ [link ↗️](https://github.com/iAmKankan/Mathematics/blob/main/LinearAlgebra/norms.md#2-manhattan-norm-or-1-norm-or--l1) If we use $L^1$ norm then

$$\Huge{\color{Purple}\Sigma^{n_x}_{i=1} |w|  =  \mathrm{\frac{\lambda}{2M}\parallel w \parallel_1}}$$


If you use **L1 regularization**, then w will end up being **sparse**. And what that means is that the **w vector** will have a **lot of zeros** in it. And some people say that this can help with **compressing the model**, because the **set of parameters are zero**, and you need **less memory** to store the model. Although, I find that, in practice, L1 regularization to make your model sparse, helps only a little bit. So I don't think it's used that much, at least not for the purpose of compressing your model. And when people train your networks, **L2 regularization** is just used much much more often.

 Lambda $\large{\color{Purple}\lambda }$ here is called the **regularization parameter** and usually, you set this using your **development set**, or using **dev set** or **cross validation**. When you a variety of values and see what does the best, in terms of trading off between doing well in your training set versus also setting that two normal of your parameters to be small. Which helps prevent **over fitting**. So **lambda** is **another hyper parameter** that you might have to tune. 

### ⚛️ $\Large{\color{Purple}\underline{\textbf{Neural Network: }}}$ 

In a **neural network**, you have a **cost function** that's a function of all of your parameters-
* $\large{\color{Purple} w^{[1]}, b^{[1]}}$ through $\large{\color{Purple} w^{[L]}, b^{[L]}}$ where capital $\large{\color{Purple} L}$ is the **number of layers** in your **neural network**. 
* And so the cost function is this, sum of the losses, summed over your $\large{\color{Purple} m}$ training examples.
*  And says at **regularization**, you add $\large{\color{Purple} \lambda / 2m}$  of sum over all of your parameters $\large{\color{Purple} w}$, 
* your parameter matrix is $\large{\color{Purple} w}$, of their, that's called the **squared norm**. 
* Where this $\large{\color{Purple} \parallel w^{(l)} \parallel^2}$ **norm of a matrix**, meaning the **squared norm** is defined as the $\large{\color{Purple} L}$,  **sum of j**, of each of the elements of that matrix, **squared**. 
* And if you want the indices of this summation. This is sum from i=1 through n[l]. Sum from j=1 through n[l-1], because $\large{\color{Purple} w}$ is $\large{\color{Purple} w: ( n^{l} ,n^{l-1})}$ dimensional matrix, where $\large{\color{Purple} n^{l} ,n^{l-1} }$ these are the number of Hidden units in layers $\large{\color{Purple}[l-1]}$ in layer $\large{\color{Purple}l}$.

So this **matrix norm**, it turns out is called the **Frobenius norm** of the matrix, denoted with a **F** in the subscript. $\large{\color{Purple}\parallel w \parallel^2_{F}}$ or $\large{\color{Purple}\parallel \dot{} \parallel_{F}^2}$

$$ \Huge{\color{Purple}\mathrm{J(w^{(1)},b^{(1)}, \cdots,w^{(\mathit{L})},b^{(\mathit{L})}) = \frac{1}{m} \Sigma^m_{i=1} L (\hat{y}^{(i)},y^{(i)})}} {\color{Cyan} + \mathrm{\frac{\lambda}{2m}\Sigma_{\mathit{l}}^L \parallel w^{(\mathit{l})} \parallel^2}}$$

$\large{\color{Purple}\textrm{Frobenius norm (Matrix norm)}}$ [link ↗️](https://github.com/iAmKankan/Mathematics/blob/main/LinearAlgebra/norms.md#frobenius-norm) **_It just means the sum of square of elements of a matrix_**

$$\Huge{\color{Purple}\mathrm{\parallel w^{(\mathit{l})} \parallel^2_{F} = \Sigma^{n^{\mathit{l}}}_{i=1} \Sigma^{n^{\mathit{l-1}}}_{j=1} \left \( w^{\mathit{l}}_{i,j}\right\)^2}}$$

$${\color{Purple}
\begin{align}
& \bullet \textrm{The limit of summation of i should be from 1 to } n^{(\mathit{l})} \\
& \bullet \textrm{The limit of summation of j should be from 1 to } n^{(\mathit{l-1})} \\
& \bullet \textrm{The rows "i" of the matrix should be the number of neurons in the current layer } n^{(\mathit{l})} \\
&\bullet \textrm{whereas the columns "j" of the weight matrix should equal the number of neurons in the previous layer } n^{(\mathit{l-1})} \\
\end{align}
}$$

$\Large{\color{Purple}\underline{\textbf{Backpropagation: }}}$ 

$$\Huge{\color{Purple}\mathrm{dw^{[\mathit{l}]} = (\textit{from backprop})}}$$

* Backprop would give us the **partial derivative** of $\large{\color{Purple}\textrm{J}}$ with respect to $\large{\color{Purple}\textrm{w}}$ or really $\large{\color{Purple}\textrm{w}}$ for any given $\large{\color{Purple}\textrm{[l]}}$ . Looks like - $\large{\color{Purple}\frac{\partial J}{\partial w^{[l]}}}$

$$\Huge{\color{Purple}\mathrm{w^{[\mathit{l}]} := \mathrm{w^{[\mathit{l}]}} - \eta \ {dw}^{[\mathit{l}]} }}$$

* And then you update w[l], as w[l]- the learning rate times d. So this is before we added this extra **regularization** term to the objective. 
* Now that we've added this **regularization** term to the objective, what you do is you take **dw** and you add to it, lambda/m times w.

$$\Huge{\color{Purple}\mathrm{dw^{[\mathit{l}]} = (\textit{from backprop})}} {\color{Cyan} + \frac{\lambda}{m} w^{[l]}}$$

*  And then you just compute this update, same as before. 
*  And it turns out that with this new definition of dw[l], this new dw[l] is still a correct definition of the derivative of your cost function, with respect to your parameters, now that you've added the extra **regularization** term at the end.**
* $\large{\color{Purple}\frac{\partial J}{\partial w^{[l]}}}{\color{Cyan}dw^{[l]} }$

And it's for this reason that L2 regularization is sometimes also called weight decay. So if I take this definition of dw[l] and just plug it in here, then you see that the update is w[l] = w[l] times the learning rate alpha times the thing from backprop, +lambda of m times w[l].

$\large{\color{Purple}\textit{Weight Decay \\# : }}$
So this is why L2 norm regularization is also called weight decay. Because it's just like the ordinally gradient descent, where you update w by subtracting alpha times the original gradient you got from backprop. But now you're also multiplying w by this thing, which is a little bit less than 1. So the alternative name for L2 regularization is weight decay. I'm not really going to use that name, but the intuition for it's called weight decay is that this first term here, is equal to this. So you're just multiplying the weight metrics by a number slightly less than 1. So that's how you implement L2 regularization in neural network.


## References:
![deep](https://user-images.githubusercontent.com/12748752/217685787-3c74eedd-9626-42ff-bab4-5e6ff825a9a4.png)
* [KNIME](https://www.knime.com/blog/regularization-for-logistic-regression-l1-l2-gauss-or-laplace)
* [Coursera](https://www.coursera.org/learn/deep-neural-network?specialization=deep-learning)
