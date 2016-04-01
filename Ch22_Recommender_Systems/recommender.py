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
from DS_Scratch.Ch4_Linear_Algebra import dot_product, magnitude

# Load Users Interests #
########################
from DS_Scratch.Ch22_Recommender_Systems.user_data import users_interests

# Popularity Recommendation #
#############################
# The easiest approach is simply to recommend what is popular
def count_interests(users_data):
    """ returns a list of tuples (interest, counts) from user_data. Assumes
    user_data is a list of list """
    return Counter(interest for user_interests in users_data for interest in
                   user_interests).most_common()

# We can suggest to a user popular interest that he/she is not already
# interested in 
def most_popular_new_interests(user_interest, users_data, max_results=5):
    """ returns unique recommendations based on most popular interests in
    user_data """
    # get the popularity of each interests
    popular_interests = count_interests(users_data)
   
    # get the recommendations
    recommendations = [(interest,count) 
                        for interest, count in popular_interests
                        if interest not in user_interest]
    # return only upto max_results
    return recommendations[:max_results]

# User-based Collaborative Filtering #
######################################
# We will use cosine similarity to measure how similar our user is to 
# other users and then use these similar users interest to make
# recommendations to  our user
def cosine_similarity(v,w):
    """ computes the normalized projection of v onto w """
    return dot_product(v,w)/float(magnitude(v)*magnitude(w))

# In order to compute the cosine similarity we will need to assign indices
# to each interest so we have user_interest_vectors to compare
# get the unique interest using set comprehension. These will be in
# alphabetical order since sort is called
def get_unique_interest(users_data):
    """ gets a sorted list of unique interest in users_data """
    return sorted(list({interest for user_interests in users_data
                        for interest in user_interests}))

# Now we want to produce an interest vector for each user. It will be binary
# a 0 for no interest and 1 for true interest at the index corresponding to
# that particular interest
def make_user_interest_vector(user_interests, users_data):
    """ makes a binary vector of interest for user """
    # get the unique interests
    unique_interests = get_unique_interest(users_data)

    return [1 if interest in user_interests else 0 
            for interest in unique_interests]

def make_user_interest_matrix(users_data):
    """ returns a matrix of all user_interest vectors """
    user_interest_matrix =[]
    for user in users_data:
        # For each user make the vector of interest
        user_interest_vector = make_user_interest_vector(user, users_data)
        # append the vector to the matrix
        user_interest_matrix.append(user_interest_vector)
    return user_interest_matrix

def user_cosine_similarities(users_data):
    """ Computes cosine simiilarity of all users in user interest matrix """
    # get the user interest matrix
    user_interest_matrix = make_user_interest_matrix(users_data)
    # compute the cosine similarity between all rows
    return [[cosine_similarity(interest_vector_i, interest_vector_j)
             for interest_vector_j in user_interest_matrix]
             for interest_vector_i in user_interest_matrix]

if __name__ == "__main__":

    # Popularity Recommendation #
    #############################

    # get the interest ordered by popularity
    popular_interests = count_interests(users_interests)
    # Interest by popularity
    #print popular_interests
    
    # print out user1 recommendations
    print most_popular_new_interests(users_interests[1], users_interests)

    # print user similarity for two sample users
    user_similarities = user_cosine_similarities(users_interests)
    print user_similarities[0][8]


