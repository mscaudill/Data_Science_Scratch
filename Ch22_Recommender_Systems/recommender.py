"""
In this module we examine how to make recommendations to users based on a
listing of each user's interest. We will do this in three ways. The first
way will simply recommend to the user what is popular across all users. The
second (more sophisticated) method will use cosine similarity between users
to recommend what other "similar users" interest are. This approach is
called 'User-based collaborative filtering'. The third approach will be to
look at the similarity in interest and generating suggestions based on the
aggregated interest. This is called item-based collaborative filtering.
"""
from collections import Counter

# Load Users Interests #
########################
from DS_Scratch.Ch22_Recommender_Systems.user_data import users_interests

# Popularity Recommendation #
#############################
# The easiest approach is simply to recommend what is popular
def count_interests(user_data):
    """ returns a list of tuples (interest, counts) from user_data. Assumes
    user_data is a list of list """
    return Counter(interest for user_interests in user_data for interest in
                   user_interests).most_common()

# We can suggest to a user popular interest that he/she is not already
# interested in 
def most_popular_new_interests(user_interest, user_data, max_results=5):
    """ returns unique recommendations based on most popular interests in
    user_data """
    # get the popularity of each interests
    popular_interests = count_interests(user_data)
   
    # get the recommendations
    recommendations = [(interest,count) 
                        for interest, count in popular_interests
                        if interest not in user_interest]
    # return only upto max_results
    return recommendations[:max_results]

# User-based Collaborative Filtering #
######################################



if __name__ == "__main__":

    # Popularity Recommendation #
    #############################

    # get the interest ordered by popularity
    popular_interests = count_interests(users_interests)
    # Interest by popularity
    #print popular_interests
    
    # print out user1 recommendations
    print most_popular_new_interests(users_interests[1], users_interests)


