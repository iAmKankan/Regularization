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
## Other Regularization Methods
![deep](https://user-images.githubusercontent.com/12748752/217685787-3c74eedd-9626-42ff-bab4-5e6ff825a9a4.png)

$\Large {\color{Purple} 1.\underline{\textrm{Data Augmentation: }}}$
If you are over **fitting**, getting more **training data** can help but getting more **training data** can be **expensive** and sometimes you just can't get more data. But what you can do is **augment** your **training set** by taking image like this. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/221146789-f38fd11a-8749-42ef-9676-e93df1fac615.png" width=90%/>
  <br>
  <ins><b>     Cat image data Augmentation  </b></ins>
</p>


1. **Flipping the image horizontally** and adding it with your **training set**.
   * So now instead of just this one example in your **training set**, you can add this to your training example. 
   * So by flipping the images horizontally, you could double the size of your training set. 
   * Because you're training set is now a bit **redundant** this isn't as good as if you had collected an additional set of **brand new independent examples**. But you could do this Without needing to pay the expense of going out to take more pictures of cats. 
2. Take **random crops of the image**. So here we're rotated and sort of **randomly zoom** into the image and this still looks like a **cat**. 
   * So by taking random **distortions** and **translations** of the image you could **augment** your data set and make additional **fake training examples**. 

Again, these extra fake training examples they don't add as much information as they were to call they get a brand new independent example of a cat. But because you can do this, almost for **free**, other than for some **computational costs**. This can be an inexpensive way to give your algorithm more data and therefore sort of **regularize it and reduce over fitting**.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/221152941-b47b0dff-13ec-4326-a1ce-956db514f4cb.png" width=70%/>
  <br>
  <ins><b> Optical Character Recognition Data Augmentation  </b></ins>
</p>


For **optical character recognition** you can also bring your data set by taking digits and imposing random rotations and distortions to it. So If you add these things to your **training set**, these are also still **digit force**. For illustration I applied a very strong distortion. So this look very wavy for, in practice you don't need to distort the four quite as aggressively, but just a more subtle distortion than what I'm showing here, to make this example clearer for you, right? But a more subtle distortion is usually used in practice, because this looks like really warped fours. So data augmentation can be used as a regularization technique, in fact similar to **regularization**.


$\Large {\color{Purple}2.\underline{\textrm{Early Stopping: }}}$
There's one other technique that is often used called **early stopping**. So what you're going to do is as you run **gradient descent** you're going to plot,
* Either the **training error**, you'll use **0, 1 classification error** on the **training set**. 
* Or just plot the **cost function** **J** **optimizing**, and that should decrease monotonically, like so, all right? Because as you trade, hopefully, you're trading around your cost function J should decrease. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/221160802-b31a4502-9cd7-4b75-94b6-4106f2a7f9fb.png" width=70%/>
  <br>
  <ins><b> Early stopping  </b></ins>
</p>

So with early stopping, what you do is you plot **Training error** and you also plot your **dev set error**. And again, this could be a classification error in a development sense, or something like the cost function, like the logistic loss or the **log loss** of the **dev set**.


<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/221164326-c24fb647-36ac-44ec-84af-14f466bd8d43.png" width=70%/>
  <br>
  <ins><b> Early stopping  </b></ins>
</p>


#### Why does this work? 
Well when you haven't have run many iterations for your neural network yet your parameters $\large{\color{Purple}\textit{w}}$ will be close to **zero** $\large{\color{Purple}\mathit{w \approx 0}}$. 
* Because with **random initialization** you probably initialize $\large{\color{Purple}\textit{w}}$ to **small random values** so before you train for a long time, $\large{\color{Purple}\textit{w}}$ is still quite small. 
* And as you iterate, as you **train**, **w** will get **bigger and bigger and bigger** until here right most side of the figure you have a much larger value of the parameters $\large{\color{Purple}\textit{w}}$ for your **neural network**. 

So what early stopping does is by stopping **halfway** you have only a **mid-size** rate $\large{\color{Purple}\textit{w}}$. And so similar to **L2 regularization** by picking a neural network with **smaller norm** for your parameters **w**, hopefully your neural network is **over fitting** less. And the term early stopping refers to the fact that you're just stopping the training of your **neural network earlier**.

$\large \textrm{Down side of Early stopping: }$ 

<p align="center">
  <img src="https://user-images.githubusercontent.com/12748752/221178537-cd344ef7-3cdd-42ed-90b6-36287e334b8e.png" width=70%/>
  <br>
  <ins><b> Early stopping  </b></ins>
</p>

The main downside of early stopping is that this couples these two tasks. So you no longer can work on these two problems independently, because by **stopping gradient decent early**, you're sort of breaking whatever you're doing to optimize cost function J, because now you're not doing a great job reducing the cost function J. You've sort of not done that that well. And then you also simultaneously trying to not over fit. So instead of using different tools to solve the two problems, you're using one that kind of mixes the two. And this just makes the set of things you could try are more complicated to think about

$\large \underline{Orthogonalization :}$

Machine learning process comprising several different steps.
1. One, is that you want an algorithm to **optimize the cost function** $\large{\color{Purple}\textit{J}}$ and we have various tools to do that, such as **Gradient Descent** , **momentum** and **RMSprop** and **Adam** and so on. 
2. But after **optimizing** the **cost function** $\large{\color{Purple}\textit{J}}$ , you also wanted to not **over-fit**. And we have some tools to do that such as your **regularization**, **getting more data** and so on. 
 * Now in machine learning, we already have so many **hyper-parameters**. It's already very complicated to choose among the space of possible algorithms. 
 * And so I find machine learning easier to think about when you have one set of tools for **optimizing** the **cost function** $\large{\color{Purple}\textit{J}}$ , and when you're focusing on **authorizing** the **cost function** $\large{\color{Purple}\textit{J}}$. 
 * All you care about is finding **w** and **b**, so that **J(w,b)** is as **small as possible**.
 *  You just don't think about anything else other than reducing this. And then it's completely separate task to not over fit, in other words, to reduce variance. And when you're doing that, you have a separate set of tools for doing it. 
 *  And this principle is sometimes called **orthogonalization**. And there's this idea, that you want to be able to think about one task at a time.
