"""

"""
from collections import Counter
from matplotlib import pyplot as plt
from prog_lang_cities import cities as city_langs

import DS_Scratch.Ch4_Linear_Algebra as Ch4

def majority_vote(labels):
    """ counts the labels and returns the label with the most counts """
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    # if there is a tie then most_common returns an arbitrary winner so
    # need to check if there was multiple winners
    num_winners = len([count for count in vote_counts.values() 
                      if count ==  winner_count])

    if num_winners == 1:
        # unique winner
        return winner
    else:
        # multiple winners, so remove a label and try again removing last
        # neighbor (label)
        return majority_vote(labels[:-1])

def knn_classifier(k, labeled_points, new_point):
    """ each labeled_point is a pair (point, label) """

    # order the points from nearest to farthest
    by_distance = sorted(labeled_points,
                         key=lambda (point,_): Ch4.distance(point,
                         new_point))

    # get the labels for the k closest point
    k_nearest_labels = [label for _, label in by_distance[:k]]

    # now we call majority vote to get a unique nearest label
    return majority_vote(k_nearest_labels)

# cities is list of tuples [(lat, long, language),...] we convert it so each
# entry is ([long, lat], favorite_language)
cities = [([longitude, latitude], language) for 
            longitude, latitude, language in city_langs] 

# plot the long/latitude of the favorite languages (see main). They seem to
# cluster so KNN seems like a good choice for a predictive model.

# Lets try to predict each cities preferred language using it's neigbors
# rather than itself (see main Sec II).

if __name__ == '__main__':

    # plot the preferred languages of each city
    plots = {'Java':([],[]),'Python':([],[]),'R':([],[])}

    for (longitude, latitude), language in cities:
         plots[language][0].append(longitude)
         plots[language][1].append(latitude)

    # create unique markers for each language
    markers = {"Java":"o", "Python":"s", "R":"^"}
    colors = {"Java":"r", "Python":"b", "R":"g"}

    # create a scatter series for each language
    for language, (x,y) in plots.iteritems():
        plt.scatter(x,y, color = colors[language], 
                    marker = markers[language],
                    label = language, zorder = 10)

    plt.legend(loc=0)
    plt.axis([-130,-60,20,55])
    plt.title("Favorite Programming Language")
    plt.show()

    # SEC II:  We will try several different values of k
    for k in [1, 3, 5, 7]:
        num_correct = 0

        for city in cities:
            location, actual_language = city
            other_cities = [other_city for other_city in cities 
                            if other_city != city]

            predicted_language = knn_classifier(k, other_cities, location)

            if predicted_language == actual_language:
                num_correct += 1

        # results of KNN of favorite languages by location
        print k, "neighbor(s):", num_correct, 'correct out of ', len(cities)


