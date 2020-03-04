# ISOMAP
## Non-Linear Dimension Reduction 

### TL;DR
ISOMAP is a non-linear dimension reduction method to reduce the size and complexity
of a dataset, projecting it onto a new plain. This method is useful for datasets
with non-linear structure, where
[PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) and [MDS](https://en.wikipedia.org/wiki/Multidimensional_scaling) will not be appropriate. 

Questions:
  - *What is this method?* ISOMAP, a Non-Linear Dimension Reduction method  
  - *What does it do? When should I use it?* This projects the dataset into a reduce
plain that shrinks the dimension while keeping _most_ of the information from
the data.  It essentially transforms the space in which the data is
represented. This can be used to shrink the size of a non-linear
high-dimensional dataset like images, or as a pre-processing step for
clustering.   
  - *What are the alternatives and how does this one compare?* Alternatives are PCA,
and MDS for linear data and t-SNE for non-linear.  There are actually many
Manifold learning methods which can be seen [here](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction) and newly published in
2018, [UMAP](https://arxiv.org/pdf/1802.03426v2.pdf)   
  - *What are it’s failure modes (when not to use) and how to you diagnose them?*
ISOMAP is dependent on thresholding or
[KNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) in it's first
step to create an adjacency matrix, which can be sensitive to noisy or sparse
data.  In addition, it can become computationally expensive.   
  - *How do I call this (is there are module/package? git repo? other?)*
Implementations of ISOMAP can found in most programming languages, but to start
exploring I suggest [Sci-Kit
Learn](https://scikit-learn.org/stable/modules/manifold.html) for Python.

### Dimension Reduction
Dimension reduction is used in when we have very high-dimensional data, which is
to say many columns or features. This appears often in image processing and 
biometric processing. In many cases, we want to reduce the complexity, size of
the data, or improve interpretation. We can use dimension reduction methods to
achieve this using methods like PCA for linear data and ISOMAP for non-linear
data.

TL;DR:
  - Feature Selection
  - Feature Projection
  - Size Reduction
  - Visual Charting in 2 or 3 dimensions 

#### When would someone use dimension reduction?
  - Under-determined system, more features than samples (Regularization)
  - Data is too large for system, cannot afford more compute
  - Reduce size of data to decrease time to solve, time sensative
  - Removal of multi-collinearity 
  - Can lead to improved model accuracy for regression and classification
    [Rico-Sulayes, Antonio (2017)](https://www.researchgate.net/publication/322835219_Reducing_Vector_Space_Dimensionality_in_Automatic_Classification_for_Authorship_Attribution)
  - Visualize the relationship between the data points when reduced to 2D or 3D

### Algorithm
There are three steps to the ISOMAP algorithm.

#### Step 1 Adjacency & Distance Matrices
Since our data has a non-linear form we will view our data as a network.  This
will allow us to have some flexibility around the shape of our data and connect
data that is "close".  There are two ways to create your edges between your
nodes 

The first way to create edges is to use
[KNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) and connect
each node to the closest `K` neighbors.  Note that we do not need to use
Euclidean distance but can and should try many different distances to see which
has the best results for your particular dataset.  Often in high dimensions,
Euclidean distance breaks down and we must start searching elsewhere. 
See links at the bottom for information on different distance metrics.

Alternatively, you can use threshold to decide your edges. If two points are
within your distance threshold, then they are connected, otherwise the distance
between the two nodes is set to infinity. You can tune your threshold so that
each node has some minimum number of connections, which is what we'll use in our
example.

Once you have your adjacency matrix, we'll create our distance matrix which has
the shortest path between each point and every other point.  This matrix will be
symmetric since we are not creating a directed graph.  This is the main
difference between ISOMAP and a linear approach.  We'll allow the path to travel
through the shape of the data showing the points are related even though they
might not actually be "close" regarding your distance metric. 

![GT ISyE 6740](assets/img/local_dist-1.png)  

Image from Georgia Tech ISyE 6740, Professor Xie

For example, if our adjacency matrix is

```
[[0   0.2 INF],
 [0.2 0   0.7]
 [INF 0.7   0]]
```

Then our distance matrix `D` would be 

```
[[0   0.2 0.9],
 [0.2 0   0.7]
 [0.9 0.7   0]]
```
Because 1 can reach 3 through 2

#### Step 2 Centering Matrix
Now we'll create the centering matrix that will be use to modify the distance
matrix `D` we just created. 

![H = I - 1/n * ee^T](assets/img/h_matrix.svg)

Now, we'll use this centering matrix on our distance matrix `D` to create our
kernel matrix, `K`

![K = -1/2HD^2H](assets/img/k_matrix.svg)

#### Step 3 Eigenvalues & Eigenvectors
Finally, we take an eigenvalue decomposition of the kernel matrix `K`. The
largest N (in our case 2) eigenvalues and their corresponding eigenvectors
are the projections of our original data into the new plain.
Since eigenvectors are all linearly independent thus, we will avoid collision.

### Example
In our example we'll take over 600 images, which are 64 x 64 and black & White,
and project them into a two dimensional space. This project can be used to
cluster or classify the images in another machine learning model.  

If we use a linear method, like PCA, we will get a reduced result, but the
relationship between the data is not clear the eye.  

![PCA Result](assets/img/pca_faces.png)

Now, by running Isomap, we can show the corresponding images by their projected
point and see that after the projection, the points that are near each other are
quite similar. So the direction the face is looking, that information was
carried through the projection.  Now this smaller dataset, from (684 x 4096) to
(684 x 2) can be used as input to a clustering method like K-means.  

![Isomap Result](assets/img/faces.png)


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

Different Distance Metrics  

  - [Intuition - Math.net](https://numerics.mathdotnet.com/Distance.html)  
  - [SciPy Distances](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)  
  - [Sci-Kit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html)  


Unrelated but [these guys](http://setosa.io/#/) in general are good.
