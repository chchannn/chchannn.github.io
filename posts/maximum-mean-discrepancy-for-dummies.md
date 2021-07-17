<!--
.. title: Maximum Mean Discrepancy for Dummies
.. slug: maximum-mean-discrepancy-for-dummies
.. date: 2021-07-16 20:24:05 UTC-07:00
.. tags: 
.. category: 
.. link: 
.. description: 
.. type: text
.. has_math: true
-->

Recently I was working on adding monitoring metrics for my text classification models to detect underlying data drift. Usually it is pretty straightforward to compare two sets of data points to check if they come from the same distribution, when working with structured, tabular inputs. There are many statistical hypothesis testings to measure the distance between two distributions for univariate. When it comes to unstructured text data as inputs, it becomes a more complicated problem. In this scenario I care about the entire vocabulary and how each word's frequency changes. One strategy to measure multivariate drift is using maximum mean discrepancy (MMD), outlined in this paper [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](https://arxiv.org/abs/1810.11953). 

Using a "simple" definition, MMD defines an idea of representing distances between distributions as distances between kernel embedding of distributions. I know this is a very confusing sentence. Here, *kernel embedding of distributions*, which has many nick names including kernel mean, feature mean, mean map, or mean embeddings of features, to simplify this mouthful jargon, is described in [wikipedia description](https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions) as
>kernel embedding of distributions comprises a class of nonparametric methods in which a probability distribution is represented as an element of a reproducing kernel Hilbert space (RKHS).

When dealing with Hilbert space, I think it is important to keep in mind that, it is simply an extension of the Euclidean space. The Euclidean space, or so called the three-dimensional spaces, or one of those corners of the room that you are in, is an orthogonal space with three perpendicular axes and planes. On the other hand, in the Hilbert space the number of orthogonal axes goes to infinity. The vector operations we like in the Euclidean space such as using inner product, which allows to define distance and angles, are also preserved in the Hilbert space.

My way to interpret kernel embedding is, like word embedding mapping a word and its semantic meaning to a set of coordinates in an orthogonal space, kernel embedding maps a distribution from domain $\Omega$ (usually non orthogonal) to a set of coordinates in an orthogonal space, so that $k(x,x')=\langle \varphi (x),\varphi (x')\rangle _{\mathcal {H}}$ can be viewed as a measure of similarity between points $x,x'\in \Omega$.
Remembering that in an orthogonal space, distance and angle measurements are more meaningful since axes are not correlated.

Empirically we can estimate kernel embedding using $$\mu_{X}=\frac{1}{n}\sum_{i=1}^{n}\varphi (x_{i}),$$ this means, if your features $X$ are orthogonal, such as using principle components after PCA, applying a liner kernel transforming features to themselves, the kernel embedding is simply the mean of each feature, or `numpy.mean(X, axis=0)`. Kernel embedding are the natural next step in that journey as it provides an orthogonal space to interpret multivariance. It also works seamlessly with dimension reduction methods like PCA.

Now, circle back to MMD, the goal is to calculate the L2 distance between two kernel embedding, aka two sets of coordinates in the Hilbert space. Recall in vector operations, the L2 distance is defined as 
<div> $$d(X,Y)^{2} = \langle X-Y,X-Y \rangle = \langle X,X \rangle + \langle Y,Y \rangle-2 \langle X,Y \rangle  $$ </div>
with vector $X, Y \in \mathbb{R}^{n}$ and their inner product as $\langle X,Y \rangle $. 
hence, MMD between $X$ in $P$ distribution and $Y$ in $Q$ distribution defines as
$$ 
MMD^{2}(P,Q)=\langle \mu_{P} , \mu_{P} \rangle -2  \langle \mu_{P} , \mu_{Q} \rangle + \langle \mu_{Q} , \mu_{Q} \rangle$$
Utilizing a property from RKHS quoting Wikipedia,
>the expectation of any function $f$ in the RKHS can be computed as an inner product with the kernel embedding.
>$$\mathbb {E} [f(X)] = \langle f,\mu_{X}\rangle_{\mathcal {H}},$$

we finally arrive a formula that could allow us to estimate MMD using similarity matrix with kernel transform in domain $\Omega$ as
$$MMD^{2}(P,Q)=E_P[k(X,X)] - 2EP_{Q}[k(X,Y)] + E_Q[k(Y,Y)]$$

In fact, many existing implementation such as [here](https://www.kaggle.com/onurtunali/maximum-mean-discrepancy#MAXIMUM-MEAN-DISCREPANCY-(MMD)-IN-MACHINE-LEARNING) or [here](https://discuss.pytorch.org/t/maximum-mean-discrepancy-mmd-and-radial-basis-function-rbf/1875) use the last formula to guide the computing process.

Once I understood the similarity matrix could be implemented as 
```
# (x - y)^2 = x^2 - 2*x*y + y^2
def similarity_matrix(mat):
    # get the product x * y
    # here, y = x.t()
    r = torch.mm(mat, mat.t())
    # get the diagonal elements
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    # compute the distance matrix
    D = diag + diag.t() - 2*r
    return D.sqrt()
```
everything else seem to be quickly pieced together.

This has been a very basic explanation to MMD, it aimed to keep things as simple as possible to describe an basic idea. MMD has extensive application in generative learning. Hopefully this article had built up enough appetite for a more detailed elaboration of the topic such as can be found in the original paper [here](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf).

