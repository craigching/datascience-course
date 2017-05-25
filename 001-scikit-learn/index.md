---
title       : Python Machine Learning
subtitle    : Introduction to scikit-learn and Jupyter
author      : Craig Ching
job         : Machine Learning Ninja
framework   : io2012        # {io2012, html5slides, shower, dzslides, ...}
highlighter : highlight.js  # {highlight.js, prettify, highlight}
hitheme     : tomorrow      # 
widgets     : [mathjax, quiz, bootstrap]            # {mathjax, quiz, bootstrap}
mode        : selfcontained # {standalone, draft}
knit        : slidify::knit2slides
---

<style>
.title-slide {
  background-color: #FFFFFF; /* #EDE0CF; ; #CA9F9D*/
}
slide:not(.segue) h2 {
  color: #800000
}
slide pre code {
  font-size: 11px ;
}
</style>

## Why Python?

* More accessible than R
    + Python is a general purpose language used for many different things including web frameworks *and* machine learning
* More popular than R
    + Maybe not with data scientists?

---

## Python for Data Science

* scikit-learn (http://scikit-learn.org)
    + Python Machine Learning
* Jupyter (http://jupyter.org)
    + Python notebooks (and other languages/environments)
* Matplotlib (http://matplotlib.org)
    + Python plotting and visualization
* Pandas (http://pandas.pydata.org)
    + Python data frames
* Others
    + Seaborn
    + Plot.ly (commercial)
    + Rodeo
    + Sympy

---

## Installation

* http://scikit-learn.org
* Requires
    + Python (>= 2.6 or >= 3.3)
    + NumPy (>= 1.6.1)
    + SciPy (>= 0.9)
    
* Installation (for Unix-like operating systems)
    + pip install numpy --upgrade
    + pip install scipy --upgrade
    + pip install scikit-learn --upgrade
    + pip install jupyter --upgrade
* Other options available at http://scikit-learn.org/stable/install.html
    + Anaconda is particularly recommended

---

## What is scikit-learn?

* Library for doing machine learning in Python
    + See main page for details
* Classification
* Regression
* Clustering
* Dimensionality reduction
* Model selection
* Preprocessing
* Reference: http://scikit-learn.org/stable/documentation.html

---

## scikit-learn map

![width](ml_map.png)

---

## What is NumPy?

* ndarray (tensor)
* Broadcasting
* Tools for integrating C/C++ and Fortran code
* Linear algebra, Fourier transform, and random number capabilities
* Reference: https://docs.scipy.org/doc/numpy/

```python
    S = my_cov(X)
    means = my_mean(X)

    # Estimate the prior from our data
    prior = X.shape[0] / np.float64(n)

    # Calculate the inverse and determinant of S for later use
    Sinv = la.inv(S)
    Sdet = la.det(S)

    # Quadratic discriminant ref Alpaydin "Introduction to Machine Learning, Third Edition"
    self.W_i = (-1/2.) * Sinv
    self.w_i = Sinv.dot(means)
    self.w_i0 = (-1/2.) * (
        means.T.dot(Sinv.dot(means))) - (1/2.) * np.log(Sdet) + np.log(prior)
```

---

## What is SciPy?

* Superset of NumPy
* Provides tools for scientific computation
* Ultimately mostly not useful to us except that it is a pre-requisite of scikit-learn
    + When implementing machine learning algorithms, there are some features that might be useful to us
* Reference: https://docs.scipy.org/doc/scipy/reference/

---

## What is Jupyter?

* A notebook, it allows you to create documents that include live code that you can share with others
* Best to see an example
    + And we will!
* Note that it used to be called "IPython"
    + The Jupyter project intends to expand beyond Python

---

## Exploring scikit-learn - Machine Learning

* Machine learning algorithms
* Classification
    + Logistic regression, SVM, Decision Trees
* Regression
    + Linear regression, SVR, Nearest Neighbors
* Clustering
    + K-means, DBSCAN, Gaussian mixtures

---

## Exploring scikit-learn - Dimensionality reduction

* Principal Component Analysis (PCA)
* Fisher's LDA
* Factor Analysis

---

## Exploring scikit-learn - Model selection

* Cross-validation
* Hyper-parameter tuning
* Model evaluation
    + Confusion matrix
    + ROC, AIC

---

## Exploring scikit-learn - Preprocessing

* Standardization
* Normalization
* Binarization
* Imputation

---

## Exploring scikit-learn - Datasets

* Boston housing prices (regression)
* Iris species (classification)
* Diabetes (regression)
* Digits (classification)
* Linnerud (multi-variate regression)

---

## scikit-learn Standard Interface

* Each machine learning algorithm (and beyond really) implements a standard interface
    + fit(X, y) - Given the data and labels, fit a model
    + predict(X) - Given unseen data, predict the response

```python
import sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## Exploring Jupyter Notebooks

* Demo

---

## Further Activities

* Install scikit-learn and Jupyter
* Download the Jupyter notebook that we worked with
* Use a decision tree to predict against the iris dataset
* Use 5-fold cross validation to report the mean error of logistic regression on the iris dataset
* Compare logistic regression and SVC on the digits dataset

---

## Pop Quiz!

* Really?!?!?

--- &radio

## Classification

scikit-learn provides many different algorithms for classification in a supervised learning setting.  Which of the following is not an algorithm that scikit-learn provides?

1. Logistic Regression
2. Support Vector Machine (SVC)
3. _Hotdog/Not Hotdog_
4. Decision Trees

*** .explanation

Logistic Regression, SVC, and Decision Trees are all algorithms provided by scikit-learn. Hotdog/Not Hotdog is a joke from the show *Silicon Valley*

--- &radio

## NumPy Matrix Calculations

NumPy, written in Python, can't possibly be efficient for numerical computing.  There are issues like the GIL, etc.  So, Python *can not* be used for efficient computing.

1. True
2. _False_

*** .explanation

To avoid the GIL and provide really efficient matrix operations, NumPy uses f2py and other native libraries to create a wrapper around lower-level Fortran and C++ code.

--- &radio

## Regression vs. Classification

C'mon, regression predicts continuous responses while classification makes discrete predictions.  There is *no way* that you can use the same algorithm for both classification and regression!

1. True
2. _False_

*** .explanation

Most algorithms do provide a way to perform both classification and regression.  Usually regression follows on from classification by using some sort of "averaging" computation.

--- &radio

Which of the following is **not** a feature of the iris dataset?

1. Sepal length
2. Petal width
3. Petal length
4. _Stem circumference_

*** .explanation

As we saw in the Jupyter notebook, the iris dataset has four features, sepal length and width and petal length and width.

--- &radio

You're on a nature hike and you see this beautiful iris along a path, but you don't know what kind it is.  You are a machine learning ninja, and you know that scikit-learn has the iris dataset, so you can predict the species if you could measure some features of the iris.  You whip out your tape measure and measure petal length and width and sepal length and width.  Now, which machine learning algorithm can help you predict what type of iris you saw?

1. k-means clustering
2. _Random Forest_
3. Ordinary least squares regression
4. DBSCAN

*** .explanation

k-means and DBSCAN are clustering algorithms, not classification algorithms.  Though you could add your measurements to the iris dataset and see where they cluster, the most appropriate alogorithm to use since you have labeled data is Random Forest.  Obviously we don't want to use linear regression for a classification problem.
