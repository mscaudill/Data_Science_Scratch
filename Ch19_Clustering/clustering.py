"""
In this module we examine k-means and hierarchal clustering. We take two
sample data sets and cluster using k m-means. This algorithm assumes:
we know the # of clusters defined in advance. It follows these steps:
1. start with a set of kmeans (a point in d-dimensions)
2. assign each point to the mean to which it is closest
3. if no point's assignment has changed stop because we've converged
4. if some points' assignment has changed recompute means and iterate
Then we examine hierarchal clustering. This algorithm merges clustersthat
are nearest neighbors until only one cluster is left. We then unmerge the
clusters in reverse order so we get k clusters. We have written but not
implemented this algorithm because it is very ineffecient. Will use the Ward
hierarchal clustering in Scipy in the future.
"""
from DS_Scratch.Ch4_Linear_Algebra import squared_distance, vector_mean

# KMeans Clustering algorithm #
###############################
class KMeans:
    """ performs k-means clustering """
    
    def __init__(self, k):
        # assign # of clusters and means to initialized clusterer object
        self.k = k
        self.means = None

    def classify(self, input):
        """ returns the index of the cluster with a vector-mean closest to
        the input """
        def helper(cluster_index):
            """ calculates the squared distance between input and
            cluster vector mean """
            return squared_distance(input, self.means[cluster_index])
                
        return min(range(self.k), key=helper)

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

# Bottom-Up Clustering Algorithm #
##################################
"""
The steps in bottom up clustering are:
1. make each input its own cluster of one
2. As long as there are multiple clusters, merge them to a cluster 
(merge_order,[leaf1,leaf2])
"""

# helper functions

def is_leaf(cluster):
    """ a cluster is a leaf if it has len of 1 """
    return len(cluster) == 1

def get_children(cluster):
    """ returns the two children of this cluster if it's a merged cluster;
     and raises exception if this is a leaf cluster """
    if is_leaf(cluster):
        raise TypeError("a leaf cluster has no children")
    else:
        return cluster[1]

def get_values(cluster):
    """ returns the value in this cluster (if its a leaf cluster) or all the
    values in the leaf clusters below it (if its not) """
    if is_leaf(cluster):
        return cluster
    else:
        return [value for child in get_children(cluster)
                          for value in get_values(child)]

def cluster_distance(cluster1, cluster2, distance_metric=min):
    """ compute all pairwise distances of points in cluster1 and cluster2
    and apply distance_metric to the resulting list """  
    return distance_metric([distance(input1, input2) 
                            for input1 in get_values(cluster1)
                            for input2 in get_values(cluster2)])

def get_merge_order(cluster):
    """ smaller numbers are later merges so when unmerging we get the
        smallest merge order first """
    if is_leaf(cluster):
        return float('inf')
    else:
        return cluster[0]

# Now create main  bottom-up algorithm
def bottom_up_cluster(inputs, distance_metric):
    # start with each input being a leaf cluster (1 tuple)
    clusters = [(input,) for input in inputs]

    # while we have more than one cluster perform the following
    while len(clusters) > 1:
        
        # compute cluster distances
        def key_helper(clusters_tuple):
            """ resturns the distance between two clusters """
            return cluster_distance(cluster_tup[1], cluster_tup[2], 
                                    distance_metric)

        # find two closest clusters
        c1, c2 = min([(cluster1, cluster2) 
                      for i, cluster1 in enumerate(clusters)
                      for cluster2 in clusters[:i]], key=key_helper)
        
        # remove them from the list of clusters since they are to be merged
        clusters = [c for c in clusters if c != c1 and c != c2]

        # merge the clusters using merge_order = # of clusters left
        merged_cluster = (len(clusters), [c1,c2])

        # add the merge to clusters
        clusters.append(merged_cluster)

    # when only one cluster left, return it
    return clusters[0]

if __name__ == '__main__':
    
    # Cluster meetups data #
    ########################
    from meet_up_data import inputs as meetups
    from matplotlib import pyplot as plt
    import matplotlib.image as mpimg
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
    image_path = r'/home/giladmeir/Pictures/4071.png'
    orig_image = mpimg.imread(image_path)

    # The image is a list of list of list where the innermost list is [r g
    # b] colors. We will take these colors and map them to 5 colors. First
    # flatten all the pixels
    pixels = [pixel for row in orig_image for pixel in row]

    # provide these to the clusterer
    im_clusterer = KMeans(5)
    im_clusterer.train(pixels)

    # now we just need to recolor each pixel with one of the 5 colors
    def recolor(pixel):
        im_cluster = im_clusterer.classify(pixel)
        return im_clusterer.means[im_cluster]
    print im_clusterer.means
    new_image = [[recolor(pixel) for pixel in row] for row in orig_image]

    fig2 = plt.figure(2, figsize=(14,4))
    ax3 = fig2.add_subplot(121)
    ax3.imshow(orig_image)
    ax3.axis('off')
    ax4 = fig2.add_subplot(122)
    ax4.imshow(new_image)
    ax4.axis('off')
     
    plt.show()

    
