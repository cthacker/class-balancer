# class-balancer
Simple package for dealing with unbalanced data sets in Python.

# Overview

I modified and simplified work from: `https://github.com/fmfn/UnbalancedDataset`, which has since
become `https://github.com/scikit-learn-contrib/imbalanced-learn` -- and is probably much better
than this code. However, that code seems to be more of a repository of techniques, while this gets the job done and does a few things they do not do.

1) This package is designed to balance classes by undersampling the over represented class *and*
oversampling the under represented group -- meeting in the middle if you will.
  - setting frac = 1 will exactly balance the classes
  - frac < 1 will make the classes closer to balanced but not all the way. This is a heuristic, that
    I found works well, in addition to modifying the data less
  - with **undersampling** first the number of data points to remove is calculated, then Tomek links
    are calculated and if enough exist will be randomly selected to be removed, else all of them
    will be removed and finally random undersampling will occur up to the number of data points to
    remove
  - with **oversampling** we use the SMOTE method and remove any Tomek links we may have
    inadvertantly created

2) The fit method generates a dictionary of weights that each class should be scaled by. This allows
for chaining oversampling and undersampling to acheive the correct class balance. 

3) You can ignore all of this and just use it as a simple black box that solves your issues.

# Installation

```
git clone https://github.com/cthacker/class-balancer.git
cd class-balancer
python setup.py install
```

# Example Use

```python
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import seaborn # not needed, makes more aesthetic plots

#import class balancer
from balancer import ClassBalancer

np.random.seed(0)
X, y = sklearn.datasets.make_moons(400, noise=0.15)

ax = plt.subplot(3, 1, 1)
ax.set_title("Original Data")
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
plt.xlim((-1.5, 2.5))
plt.ylim((-1.5, 1.5))

# remove 70% of class 0
delmask = []
for i, cl in enumerate(y):
    if cl == 0:
        if np.random.random_integers(1, 10) > 3:
            delmask.append(i)

X = np.delete(X, delmask, axis=0)
y = np.delete(y, delmask, axis=0)
ax = plt.subplot(3, 1, 2)
ax.set_title("Red data has been artificially reduced")
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
plt.xlim((-1.5, 2.5))
plt.ylim((-1.5, 1.5))


newclass = ClassBalancer(random_state=0, verbose=True, frac=1.0)
newx, newy = newclass.fit_transform(X, y)
ax = plt.subplot(3, 1, 3)
ax.set_title("Classes are now balanced")
plt.scatter(newx[:, 0], newx[:, 1], s=40, c=newy, cmap=plt.cm.Spectral)
plt.xlim((-1.5, 2.5))
plt.ylim((-1.5, 1.5))

plt.show()  
```

**Output**

```
Determining class statistics...
2 classes detected: {0: 52, 1: 200} with weights:  {0: 2.4230769230769229, 1: 0.63}
Start resampling ...
3 Tomek links found.
Under-sampling performed, removed tomek links: Counter({1: 197, 0: 52})
Under-sampling performed, new total: Counter({1: 126, 0: 52})
Determining class statistics...
2 classes detected: {0: 52, 1: 126} with weights:  {0: 2.4230769230769229, 1: 0.63}
Generated 73 new samples ...
Over-sampling performed: Counter({1.0: 126, 0.0: 125})
```

![Class Balancer on Moon data set](/example/balanced_data.png?raw=true "Balanced Data")
