"""
Based off of: https://github.com/fmfn/UnbalancedDataset
Now handles more than 2 classes

only implements some of that code
Oversampling: Smote + Tomek Links
Undersampling: Tomek Links + Random

Adds a fraction to determine how balanced the data will become
Also allows for passing in class weights
"""

from collections import Counter
from copy import deepcopy
from sklearn.neighbors import NearestNeighbors
import numpy as np


class UnbalancedDataset(object):
    # pylint: disable=no-member
    """
    Parent class with the main methods: fit, transform and fit_transform
    """

    def __init__(self, random_state=None, verbose=True, frac=0.8):
        """
        Initialize this object and its instance variables.
        :param random_state:
            Seed for random number generation.
        :param verbose:
            Boolean to either or not print information about the processing
        :return:
            Nothing.
        Instance variables:
        -------------------
        :self.rs:
            Holds the seed for random state
        :self.x:
            Holds the feature matrix.
        :self.y:
            Holds the target vector.
        :self.weight_dict:
            Dictionary holding weights for each class to be scaled by
        :self.ucd:
            Dictionary to hold the label of all the class and the number of
            elements in each.
            {'label A' : #a, 'label B' : #b, ...}
        :self.frac:
            Allows making the classes balanced, but not completely so
        """

        self.rs = random_state
        self.x = None
        self.y = None
        self.weight_dict = {}
        self.ucd = {}
        self.out_x = None
        self.out_y = None
        self.verbose = verbose
        self.frac = frac

    def resample(self):
        pass

    def fit(self, x, y, provide_class_weight=None):
        """
        Class method to find the relevant class statistics and store it.
        :param x:
            Features.
        :param y:
            Target values.
        :return:
            Dictionary of weights
        """

        self.x = x
        self.y = y

        if self.verbose:
            print "Determining class statistics... "

        # Get all the unique elements in the target array and counts
        uniques = set(self.y)
        class_counts = np.bincount(self.y)

        # keep track of labels
        class_labels = []
        for elem in uniques:
            class_labels.append(elem)

        # Create a dictionary to store the statistic for each element
        for i, count in enumerate(class_counts):
            self.ucd[class_labels[i]] = count

        # something#
        if len(uniques) == 1:
            raise RuntimeError("Only one class detected, aborting...")

        # get weight array -- don't want to fully make classes equal, just closer
        if provide_class_weight is not None:
            self.weight_dict = provide_class_weight
        else:
            weight_array = len(self.y)/(float(len(uniques)) * class_counts)
            for i in xrange(len(weight_array)):
                # limits how much undersampling/oversampling we do by applying the frac
                if weight_array[i] < 1:
                    weight_array[i] = 1 - (1 - weight_array[i])*self.frac
                else:
                    weight_array[i] = (weight_array[i] - 1.)*self.frac + 1

            # create weight dict to store new sample weights
            for i, weight in enumerate(weight_array):
                self.weight_dict[class_labels[i]] = weight

        if self.verbose:
            print str(len(uniques)) + \
                " classes detected: " + \
                str(self.ucd) + " with weights: ", \
                str(self.weight_dict)

        return self.weight_dict

    def transform(self):
        """
        Class method to re-sample the dataset with a particular technique.
        :return:
            The re-sampled data set.
        """
        if self.verbose:
            print("Start resampling ...")

        self.out_x, self.out_y = self.resample()

        return self.out_x, self.out_y

    def fit_transform(self, x, y, provide_class_weight=None):
        """
        Class method to fit and transform the data set automatically.
        :param x:
            Features.
        :param y:
            Target values.
        :return:
            The re-sampled data set.
        """

        self.fit(x, y, provide_class_weight)
        if provide_class_weight is not None:
            self.weight_dict = provide_class_weight

        self.out_x, self.out_y = self.resample()

        return self.out_x, self.out_y

    @staticmethod
    def is_tomek(y, nn_index, class_type, complement=False, verbose=True):
        # pylint: disable=simplifiable-if-statement
        """
        is_tomek uses the target vector and the first neighbour of every sample
        point and looks for Tomek pairs. Returning a boolean vector with True
        for majority Tomek links.
        :param y:
            Target vector of the data set, necessary to keep track of whether a
            sample belongs to minority or not
        :param nn_index:
            The index of the closes nearest neighbour to a sample point.
        :param class_type:
            The label of the minority class.
        :param complement:
            If False will do normal behavior and remove tomek links from classes that aren't
            class_type.
            If True adds the class of interest data point as a tomek link to be removed.
        :param verbose:
            Boolean controlling if information about tomek links is printed or not
        :return:
            Boolean vector on len( # samples ), with True for majority samples
            that are Tomek links.
        """

        # Initialize the boolean result as false, and also a counter
        links = np.zeros(len(y), dtype=bool)
        count = 0

        # Loop through each sample and looks whether it belongs to the class of interest. If it
        # does, we don't consider it since we are interested in data points that have this class as
        # a neighbor. If, however, it belongs to another class we look at its first neighbour. If
        # its closest neighbour also has the current sample as its closest neighbour, the two form a
        # Tomek link.
        for ind, ele in enumerate(y):
            if ele == class_type:
                continue

            if y[nn_index[ind]] == class_type:

                # If they form a tomek link, put a True marker on this
                # sample, and increase counter by one.
                if nn_index[nn_index[ind]] == ind:
                    # remove the point from the class of interest
                    if complement:
                        links[nn_index[ind]] = True
                    # else remove from the other class
                    else:
                        links[ind] = True
                    count += 1

        if verbose:
            print("%i Tomek links found." % count)

        return links

    @staticmethod
    def make_samples(x, nn_data, y_type, knns, n_samples,
                     step_size=1., random_state=None, verbose=True):
        """
        A support function that returns artificial samples constructed along
        the line connecting nearest neighbours.
        :param x:
            Minority points for which new samples are going to be created.
        :param nn_data:
            Data set carrying all the neighbours to be used
        :param y_type:
            The minority target value, just so the function can return the
            target values for the synthetic variables with correct length in
            a clear format
        :param knns:
            The nearest neighbours to be used.
        :param n_samples:
            The number of synthetic samples to create.
        :param random_state:
            Seed for random number generation.
        :return:
            new: Synthetically generated samples.
            y_new: Target values for synthetic samples.
        """

        # A matrix to store the synthetic samples
        new = np.zeros((n_samples, len(x.T)))

        # Set seeds
        np.random.seed(random_state)
        seeds = np.random.randint(low=0,
                                  high=100*len(knns.flatten()),
                                  size=n_samples)

        # Randomly pick samples to construct neighbours from
        np.random.seed(random_state)
        samples = np.random.randint(low=0,
                                    high=len(knns.flatten()),
                                    size=n_samples)

        # Loop over the NN matrix and create new samples
        for i, n in enumerate(samples):
            # NN lines relate to original sample, columns to its
            # nearest neighbours
            row, col = divmod(n, len(knns.T))

            # Take a step of random size (0,1) in the direction of the
            # n nearest neighbours
            np.random.seed(seeds[i])
            step = step_size * np.random.uniform()

            # Construct synthetic sample
            new[i] = x[row] - step * (x[row] - nn_data[knns[row, col]])

        # The returned target vector is simply a repetition of the
        # minority label
        y_new = np.ones(len(new)) * y_type

        if verbose:
            print("Generated %i new samples ..." % len(new))

        return new, y_new


class RandUnderSample(UnbalancedDataset):
    # pylint: disable=no-member
    """
    An implementation of Tomek + Random undersampling.
    """

    def __init__(self, tomek=True, complement=True, random_state=None,
                 replacement=False, verbose=True, frac=0.8):
        """
        :param random_state:
            Seed.
        :return:
            ret_x, ret_y: The undersampled data
        """

        UnbalancedDataset.__init__(self,
                                   random_state=random_state,
                                   verbose=verbose,
                                   frac=frac)

        # Instance variable to store the number of neighbours to use.
        self.tomek = tomek
        self.complement = complement
        self.replacement = replacement

    def resample(self):
        # loop over all classes that need oversampling
        ret_x = deepcopy(self.x)
        ret_y = deepcopy(self.y)
        for class_id, weight in self.weight_dict.iteritems():
            if weight < 1:
                len_class = len(self.y[self.y == class_id])

                # number of samples to remove
                n_samples = int(len_class - weight*len_class)

                # remove tomek links first
                if self.tomek:
                    # Find the nearest neighbour of every point
                    nn = NearestNeighbors(n_neighbors=2)
                    nn.fit(ret_x)
                    nns = nn.kneighbors(ret_x, return_distance=False)[:, 1]

                    # Send the information to is_tomek function to get boolean vector back
                    # links = self.is_tomek(sy, nns, class_id,
                    links = self.is_tomek(ret_y, nns, class_id,
                                          complement=self.complement,
                                          verbose=self.verbose)
                    true_links = links[links]
                    if len(true_links) > n_samples:
                        print "I'm here"
                        # random sample the tomek links until it equals the number desired samples
                        to_remove = len(true_links) - n_samples
                        ind_to_flip = np.random.choice(np.where(links)[0],  # nopep8
                                                       to_remove,
                                                       replace=False)
                        links[ind_to_flip] = False
                        if self.verbose:
                            print("Under-sampling performed, removed tomek links:"
                                  " " + str(Counter(ret_y[np.logical_not(links)])))
                        ret_x = ret_x[np.logical_not(links)]
                        ret_y = ret_y[np.logical_not(links)]
                    else:
                        # since we removed tomek links, subtract from n_samples needed to remove
                        n_samples -= len(true_links)
                        if self.verbose:
                            print("Under-sampling performed, removed tomek links:"
                                  " " + str(Counter(ret_y[np.logical_not(links)])))
                        # remove the links
                        ret_x = ret_x[np.logical_not(links)]
                        ret_y = ret_y[np.logical_not(links)]

                # remove data by random sampling
                # get indices we are going to remove
                ind_to_remove = np.random.choice(np.where(ret_y == class_id)[0],
                                                 n_samples,
                                                 replace=False)
                ret_x = np.delete(ret_x, ind_to_remove, axis=0)
                ret_y = np.delete(ret_y, ind_to_remove, axis=0)
                if self.verbose:
                    print("Under-sampling performed, new total:"
                          " " + str(Counter(ret_y)))

        return ret_x, ret_y


class Smote(UnbalancedDataset):
    # pylint: disable=no-member
    """
    An implementation of SMOTE + Tomek.
    Comparison performed in "Balancing training data for automated annotation
    of keywords: a case study", Batista et al. for more details.
    """

    def __init__(self, k=5, tomek=False, complement=False,
                 random_state=None, verbose=True, frac=0.8):
        """
        :param k:
            Number of nearest neighbours to use when constructing the
            synthetic samples.
        :param random_state:
            Seed.
        :return:
            The resampled data set with synthetic samples concatenated at the
            end.
        """

        UnbalancedDataset.__init__(self,
                                   random_state=random_state,
                                   verbose=verbose,
                                   frac=frac)

        # Instance variable to store the number of neighbours to use.
        self.k = k
        self.tomek = tomek
        self.complement = complement

    def resample(self):
        # loop over all classes that need oversampling
        ret_x = deepcopy(self.x)
        ret_y = deepcopy(self.y)
        for class_id, weight in self.weight_dict.iteritems():
            if weight > 1:
                minx = self.x[self.y == class_id]
                miny = self.y[self.y == class_id]

                # number of samples to create
                n_samples = int(weight*len(miny) - len(miny))

                # Finding nns
                nearest_neighbour = NearestNeighbors(n_neighbors=self.k + 1)
                nearest_neighbour.fit(minx)
                knns = nearest_neighbour.kneighbors(minx, return_distance=False)[:, 1:]

                # Creating synthetic samples
                sx, sy = self.make_samples(minx,
                                           minx,
                                           class_id,
                                           knns,
                                           n_samples,
                                           random_state=self.rs,
                                           verbose=self.verbose)

                # remove tomek links created by oversampling only -- don't remove real data
                if self.tomek:
                    # Find the nearest neighbour of every point
                    nn = NearestNeighbors(n_neighbors=2)
                    ret_x = np.concatenate((ret_x, sx), axis=0)
                    ret_y = np.concatenate((ret_y, sy), axis=0)
                    nn.fit(ret_x)
                    nns = nn.kneighbors(ret_x, return_distance=False)[:, 1]

                    # Send the information to is_tomek function to get boolean vector back
                    links = self.is_tomek(ret_y, nns, class_id,
                                          complement=self.complement,
                                          verbose=self.verbose)
                    ret_x = ret_x[np.logical_not(links)]
                    ret_y = ret_y[np.logical_not(links)]
                    if self.verbose:
                        print("Over-sampling performed with Tomek links removed:"
                              " " + str(Counter(ret_y)))

                else:
                    if self.verbose:
                        print("Over-sampling performed:"
                              " " + str(Counter(np.concatenate((ret_y, sy)))))

                    # Concatenate the newly generated samples to the original data set
                    ret_x = np.concatenate((ret_x, sx), axis=0)
                    ret_y = np.concatenate((ret_y, sy), axis=0)

        return ret_x, ret_y


class ClassBalancer(object):
    """
    Performs undersampling majority, then oversampling minority

    Currently expects y to be list of integers starting with zero
    """

    def __init__(self, random_state=0, verbose=False, frac=0.8):
        self.random_state = random_state
        self.verbose = verbose
        self.frac = frac

    def fit_transform(self, x, y):
        # fit first so they have the same weight matrix
        undersample = RandUnderSample(random_state=self.random_state, verbose=self.verbose,
                                      frac=self.frac)
        oversample = Smote(random_state=self.random_state, verbose=self.verbose,
                           frac=self.frac)
        provide_class_weight = undersample.fit(x, y)

        # undersample first removing tomek links
        newx, newy = undersample.transform()
        newx, newy = oversample.fit_transform(newx, newy, provide_class_weight)
        return newx, newy
