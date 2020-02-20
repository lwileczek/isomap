# ISOMAP
## Non-Linear Model Reduction 

### TL;DR
ISOMAP is a non-linear model reduction method to reduce the size and complexity
of a dataset, projecting it onto a new plain. This method is useful for datasets
with non-linear structure, where
[PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) and [MDS](https://en.wikipedia.org/wiki/Multidimensional_scaling) will not be appropriate. 

*What is this method?* ISOMAP, a Non-Linear Dimension Reduction method  
*What does it do? When should I use it?* This projects the dataset into a reduce
plain that shrinks the dimension while keeping _most_ of the information from
the data.  It essentially transforms the space in which the data is
represented. This can be used to shrink the size of a non-linear
high-dimensional dataset like images, or as a pre-processing step for
clustering.   
*What are the alternatives and how does this one compare?* Alternatives are PCA,
and MDS for linear data and t-SNE for non-linear.  There are actually many
Manifold learning methods which can be seen [here](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction) and newly published in
2018, [UMAP](https://arxiv.org/pdf/1802.03426v2.pdf)   
*What are it’s failure modes (when not to use) and how to you diagnose them?*
ISOMAP is dependent on thresholding or
[KNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) in it's first
step to create an adjacency matrix, which can be sensitive to noisy or sparse
data.  In addition, it can become computationally expensive.   
*How do I call this (is there are module/package? git repo? other?)*
Implementations of ISOMAP can found in most programming languages, but to start
exploring I suggest [Sci-Kit
Learn](https://scikit-learn.org/stable/modules/manifold.html) for Python.

### Dimension Reduction
Dimension reduction is used in when we have very high-dimensional data, which is
to say many columns or features. This often appears with image processing or
biometric processing. In many cases, we want to reduce the complexity, size of
the data, or improve interpretation. We can use dimension reduction methods to
achieve this using methods like PCA for linear data and ISOMAP for non-linear
data.

### Algorithm
There are three steps to the ISOMAP algorithm.

#### Step 1 Adjacency & Distance Matrices

### Results
Show results compared to PCA

### Additional Resources

  - [Sci-kit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html)
  - [Original Paper](https://web.mit.edu/cocosci/Papers/sci_reprint.pdf)
  - [Another Example](https://towardsdatascience.com/decomposing-non-linearity-with-isomap-32cf1e95a483)  
  - [t-SNE](https://en.wikipedia.org/wiki/Dimensionality_reduction) A great
    alternative 
  - [More t-SNE](https://distill.pub/2016/misread-tsne/) Interactive visuals   

Wiki articles:  
  - [Nonlinear Dimensionality reduction](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction)
  - [Model Reduction](https://en.wikipedia.org/wiki/Model_order_reduction)
  - [Dimensionality Reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction)


Unrelated but [these guys](http://setosa.io/#/) in general are good.
