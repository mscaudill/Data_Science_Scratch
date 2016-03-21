"""
# of clusters defined in advance
1. start with a set of kmeans (a point in d-dimensions)
2. assign each point to the mean to which it is closest
3. if no point's assignment has changed stop because we've converged
4. if some points' assignment has changed recompute means and iterate
"""
from DS_Scratch.Ch4_Linear_Algebra import squared_distance, vector_mean

class KMeans(object):
    """ performs k-means clustering """
    def __init__(self, k):
        # assign # of clusters and means to initialized clusterer object
        self.k = k
        self.means = None

    def classify(self, input):
        """ returns the index of the cluster with a vector-mean closest to
            the input"""
        return min(range(self.k), 
                    key=lambda i: squared_distance(input, self.means[i]))

    def train(self, inputs):
        # Choose k random points as the initial means
        self.means = random.sample(inputs, self.k)
        assignments = None

        while True:
            # Find new assignments
            new_assignments = map(self.classify, inputs)
            
            # if assignments have changed we are done
            if assignments == new_assignments:
                return

            # Otherwise keep the new assignments
            assignments = new_assignments

            # and compute the new means
            for i in range(self.k):
                # get all the points assigned to this cluster
                i_points = [p for p,a in zip(inputs,assignments) if a == i]

                # make sure i_points is not empty so we don't divide by 0
                if i_points:
                    self.means[i] = vector_mean(i_points)
                    

if __name__ == '__main__':
    
    # Cluster meetups data #
    ########################
    from meet_up_data import inputs as meetups
    from matplotlib import pyplot as plt
    import random

    xs = [ls[0] for ls in meetups]
    ys = [ls[1] for ls in meetups]
    
    fig = plt.figure(1, figsize=(14,6))
    ax = fig.add_subplot(121)

    ax.scatter(xs,ys)
    ax.set_title('Location Data')
    ax.set_xlabel('Blocks East of city center')
    ax.set_ylabel('Blocks North of city center')
    
    # compute the means for the three clusters
    random.seed(0)
    clusterer = KMeans(3)
    clusterer.train(meetups)
    # add the means to the plot
    for cluster_index, mean in enumerate(clusterer.means):
        ax.text(mean[0], mean[1], str(cluster_index + 1))
    
    # Choosing K #
    ##############
    def squared_clustering_errors(inputs, k):
        """ finds total squared error from k-means clustering of inputs """
        clusterer = KMeans(k)
        clusterer.train(inputs)
        means = clusterer.means
        assignments = map(clusterer.classify, inputs)

        return sum(squared_distance(input, means[cluster])
                   for input, cluster in zip(inputs, assignments))

    ks = range(1, len(meetups) + 1)
    errors = [squared_clustering_errors(meetups, k) for k in ks]

    ax2 = fig.add_subplot(122)
    ax2.plot(ks, errors)
    ax2.set_xlabel('k number of clusters')
    ax2.set_ylabel('Total squared error')
    ax2.set_title('Total Error vs. Number of Clusters')
    
    # Clustering Colors #
    #####################

    
    
    
    
    
    
    plt.show()

    
