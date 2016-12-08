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

# Example Use




