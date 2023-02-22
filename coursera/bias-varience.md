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

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/220549436-d54717c0-36c7-44af-a9c5-ac3014057755.png" width=70%/>
  <br>
  <ins><b>     Bias - Varience  </b></ins>
</p>


Let's say the data set that looks like this. 

$\Large\textrm{Left most pic: }$ If you fit a straight line to the data, maybe get a **logistic regression** fit to that. This is not a very good fit to the data. And so this is class of a **high bias**, what we say that this is **underfitting the data**. 

$\Large\textrm{Right most pic: }$
On the opposite end, if you fit an incredibly complex classifier, maybe deep neural network, or neural network with all the hidden units, maybe you can fit the data perfectly, but that doesn't look like a great fit either. So there's a classifier of high variance and this is **overfitting the data**. 

And there might be some classifier in between, with a medium level of complexity, that maybe fits it correctly like that. That looks like a much more reasonable fit to the data, so we call that just right. It's somewhere in between. 

In high dimensional problems, you can't plot the data and visualize division boundary. Instead, there are couple of different metrics, that we'll look at, to try to understand bias and variance.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/220562468-ec04b922-3e2f-44b1-ba51-cbd1652834d2.png" width=80%/>
  <br>
  <ins><b>     Bias / Variance with Example </b></ins>
</p>

Example of **cat picture classification**: where $\large{\color{Purple}\textit{Y=1}}$ is a **positive example** and $\large{\color{Purple}\textit{Y=0}}$ is a **negative example**, the two key numbers to look at to understand **bias** and **variance** will be **the train set error** and **the dev set** or **the development set error**. 

$\large{\color{Purple}\textit{Example \\#1: }}$ So let's say, your **training set error** is **1%** and your **dev set error** is **11%**. 
* So in this example, you're **doing very well on the training set**, 
* But you're **doing relatively poorly on the development set**.
* So this looks like you might have **overfit the training set**, that somehow you're **not generalizing well**, to this whole **cross-validation set** in the development set.
*  And so if you have an example like this, we would say this has **high variance**. 

So by looking at the training set error and the development set error, you would be able to render a diagnosis of **your algorithm having high variance**.

$\large{\color{Purple}\textit{Example \\#2: }}$ Now, let's say, that you measure your **training set** and your **dev set error**, and you get a different result. Let's say, that your **training set error is 15%** and your **dev set error is 16%**. 
* In this case, assuming that humans achieve roughly 0% error, that humans can look at these pictures and just tell if it's cat or not, then it looks like **the algorithm is not even doing very well** on **the training set**. 
* So if it's not even fitting the training data seam that well, then this is **underfitting the data**. And so this **algorithm has high bias**. 
* But in contrast, this actually **generalizing** at a **reasonable level** to the **dev set**, whereas performance in the **dev set** is only **1%** worse than performance in the **training set**. 
* So this algorithm has a problem of **high bias**, because it was not even fitting the **training set**. Well, this is similar to the leftmost plots we had on the previous slide.

$\large{\color{Purple}\textit{Example \\#3: }}$ Now, here's another example. Let's say that you have **15%** **training set error**, so that's **pretty high bias**, but when you evaluate to the **dev set** it does even worse, maybe it does **30%** 
* In this case, I would diagnose this algorithm as having **high bias**, because it's not doing that well on the **training set**, and **high variance**. 

So this has really the worst of both worlds.

$\large{\color{Purple}\textit{Example \\#4: }}$ If you have **0.5** **training set error**, and **1% dev set error**.
* That you have a **cat classifier** with only **1%**, than just we have **low bias** and **low variance**. 

This analysis is predicated on the assumption, that human level performance gets nearly **0%** error or, more generally, that the **optimal error**, sometimes called **bayes error**, so the **bayesian optimal error nearly 0%**. 

If the **optimal error** or the **bayes error** were much higher, say, it were **15%**, then if you look at this **classifier**, **15%** is actually perfectly reasonable for training set and you wouldn't see it as **high bias** and also a **pretty low variance**. 

$\Large\underline{\textrm{High Bias and High Variance: }}$
#### What does high bias and high variance looks like? 

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/220582688-939a201c-4be8-4ab9-872e-e65d6abb26cf.png" width=40%/>
  <img src="https://user-images.githubusercontent.com/12748752/220583220-299c3496-12da-42b3-8459-045fa7b1ff99.png" width=40%/>
  <br>
  <ins><b>     Bias / Variance with Example </b></ins>
</p>

This is kind of the worst of both worlds. 
* **Left:** A classifier like this, then your classifier has high bias, because it **underfits** the data. So this would be a classifier that is mostly linear and therefore **underfits** the data, we're drawing this is purple. 
* But if somehow your classifier does some weird things, then it is actually **overfitting** parts of the data as well. 
* So the classifier that I drew in purple, has both **high bias** and **high variance**.

Where it has high bias, because, by being a mostly linear classifier, is just not fitting. You know, this quadratic line shape that well, but by having too much flexibility in the middle, it somehow gets this example, and this example overfits those two examples as well. 

* So this classifier kind of has **high bias** because it was **mostly linear**, but you need maybe a **curve function** or **quadratic function**.
* And it has **high variance**, because it had too much **flexibility to fit those two mislabel**, or those live examples in the middle as well. 

In case this seems contrived, well, this example is a little bit contrived in two dimensions, but with very high dimensional inputs. You actually do get things with high bias in some regions and high variance in some regions, and so it is possible to get classifiers like this in high dimensional inputs that seem less contrived.




