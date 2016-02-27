"""

In this module we examine how to use k nearest neighbors model to predict a
city's preferred programming language by looking at the cities k nearest
neighbors (based on longitude and latitude).

"""
from collections import Counter
from matplotlib import pyplot as plt
from prog_lang_cities import cities as city_langs
from plot_state_bounds import plot_states

import DS_Scratch.Ch4_Linear_Algebra as Ch4

def majority_label(labels):
    """ counts the labels and returns the label with the most counts """

    # count the number of times a label appears
    label_counts = Counter(labels)
    # get the winner of the label_counts
    winner, winner_count = label_counts.most_common(1)[0]
    # if there is a tie then most_common returns an arbitrary winner so
    # need to check if there was multiple winners
    num_winners = len([count for count in label_counts.values() 
                      if count ==  winner_count])

    if num_winners == 1:
        # unique winner
        return winner
    else:
        # multiple winners, so remove a label and try again removing last
        # neighbor (label)
        return majority_label(labels[:-1])

def knn_classifier(k, labeled_points, new_point):
    """ returns the majority label for the k-nearest points around new 
        point.Each labeled_point is a pair (point, label) """
    
    def new_point_distance(labeled_point):
        """ helper function to calculate distance between a (point, label)
            tuple and new_point """
        # get the location from the labeled_point
        point = labeled_point[0]
        # return distance
        return Ch4.distance(point, new_point)

    # order the points from nearest to farthest
    by_distance = sorted(labeled_points, key = new_point_distance)

    # get the labels for the k closest point
    k_nearest_labels = [label for _, label in by_distance[:k]]

    # now we call majority vote to get a unique nearest label
    return majority_label(k_nearest_labels)


if __name__ == '__main__':


    """ Here we plot the preferred languages of each city with different
    markers to see how they cluster. Its a good start to see if the KNN
    model is applicable. Are there clusters? Yes but they appear weak """
    
    # Preferred language by city. Data taken from prog_lang_cities.py
    # cities is list of tuples [(lat, long, language),...] we convert it 
    # so each entry is ([long, lat], favorite_language)
    cities = [([longitude, latitude], language) for 
                longitude, latitude, language in city_langs] 
   
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
    # call the plot_states function from the plot_states_boundaries mod.
    plot_states(plt)

    """ In this section, we will try to predict for each city what its
    preferred programming language is by looking at the preferred
    programming language label from the k nearest cities. We do this for
    several k neighbor values to find the best one """

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

    """ In this section we will look at how regions of the country would
    classify rather than just the cities. We do this by making a grid over
    the latitudes and longitudes and comparing that (long,lat) with the
    nearest k cities preferred language"""

    def regional_programming_languages(k):
        """ Returns the knn classification of latitude, longitude positions
            using the labels from the k nearest cities"""
        # get the preferred languages of each city
        regional_plots = {'Java':([],[]),'Python':([],[]),'R':([],[])}

        for longitude in range(-130,-60):
            for latitude in range(20, 55):
                predicted_language = knn_classifier(k, cities,
                                                    [longitude, latitude])

                regional_plots[predicted_language][0].append(longitude)
                regional_plots[predicted_language][1].append(latitude)

        return regional_plots

    # Plot for k = 3
    regional_plots = regional_programming_languages(3)
    # create a scatter series for each language
    plt.figure(2)
    for language, (x,y) in regional_plots.iteritems():
        plt.scatter(x,y, color = colors[language], 
                        marker = markers[language],
                        label = language, zorder = 10)

    l=plt.legend(loc=0)
    l.set_zorder(20)

    plt.axis([-130,-60,20,55])
    plt.title("Favorite Programming Language By Region")
    # call the plot_states function from the plot state bounds mod
    plot_states(plt,color = 'black')
    plt.show()
        

