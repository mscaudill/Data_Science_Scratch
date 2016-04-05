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
from collections import Counter, defaultdict
from DS_Scratch.Ch4_Linear_Algebra import dot_product, magnitude
from operator import itemgetter

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

def most_similar_users_to(user_id):
    """ sorts the users based on cosine similarity to the user with
    user_id """
    # get all possible pair partners to user_id. exclude those with 0
    # similarity

    # first get the cosine similarities between all users
    similarities =  user_cosine_similarities(users_interests)
    # now get all possible non-zero pairs
    pairs = [(other_user_id, similarity) 
              for other_user_id, similarity in
                enumerate(similarities[user_id]) 
              if user_id != other_user_id and similarity > 0]
    # tuple sorting is by first el first which is other_user_id so we must
    # sort by 2nd el of tuple
    return sorted(pairs, key=itemgetter(1), reverse=True)

def user_based_suggestions(user_id, include_current_interests=False):
    """ makes suggestions to user with id user_id based on their cosine
    similarity with other users """
    suggestions = defaultdict(float)
    # go through the tuple list of user_ids and similarity of most_similar
    # users_to and pool the interest of each other_user_id into a defdict
    for other_user_id, similarity in most_similar_users_to(user_id):
        for interest in users_interests[other_user_id]:
            suggestions[interest] += similarity

    # convert the interest into a sorted list sorting by the number of times
    # that suggestion occurs
    suggestions = sorted(suggestions.items(), key=itemgetter(1), 
                         reverse=True)

    # possibly exclude from suggestions interest that already belong to
    # user_ids interest list
    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight) for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]

# Item-Based Collaborative Filtering #
######################################
# As the number of possible interest increases (i.e. the num dimensions of
# our vector increases) it becomes less and less likely that any two vectors
# will be very similar. In high D spaces vectors tend to be very far apart
# (see curse of dimensionality sec). So another method for making
# suggestions to users is to do item-based collaborative filtering. In this
# approach we compute the similarity between interests rather than users and
# make recommendations from a pool of similar interest.

# first we will transpose the user_interests_matrix so that rows correspond
# to interests and cols correspond to users
def transpose_user_interests():
    # make the user_interests matrix
    user_interests_matrix = make_user_interest_matrix(users_interests)
    # get the unique interests
    unique_interests = get_unique_interest(users_interests)
    # perform the transpose so that now we have a matrix where each row is
    # an interest and the cols are a 0 or 1 for each users index
    interest_user_matrix = [[user_interest_vector[j]
                        for user_interest_vector in user_interests_matrix]
                        for j,_ in enumerate(unique_interests)]

    return interest_user_matrix

# now we compute the similarities between the interest
def interests_similarity():
    """ computes the similarity between interests """
    # first get the interest user matrix
    interests_user_matrix = transpose_user_interests()

    # compute the cosine similarities between the interest vectors
    interest_similarities = [
                        [cosine_similarity(user_vector_i, user_vector_j)
                        for user_vector_j in interests_user_matrix]
                        for user_vector_i in interests_user_matrix]
    
    return interest_similarities

# now we can find the most similar interest to each interest with
def most_similar_interest_to(interest_id):
    """ orders the interest in terms of cosine similarity to interest of
    interest_id """

    # Get the interest similarities and pull ot the list for the interest we
    # want (interest_id)
    interest_similarities = interests_similarity()
    similarities = interest_similarities[interest_id]

    # get the unique interests
    unique_interests = get_unique_interest(users_interests) 
    
    # get the pairs of unique interest and similarity 
    pairs = [(unique_interests[other_interest_id], similarity)
              for other_interest_id, similarity in enumerate(similarities)
              if interest_id != other_interest_id and similarity > 0]
    
    # return the sorted tuples from largest to smallest similarity
    return sorted(pairs,key=itemgetter(1), reverse=True)

# Now that we have a list and rank of interest similar to a given interest
# we can make recommendations to the user 
def item_based_recommendations(user_id, include_current_interest=False):
    """ uses the users interest to determine similar interest to make
    recommendation """

    suggestions = defaultdict(float)
    # get the user_interest_matrix
    user_interest_matrix = make_user_interest_matrix(users_interests)
    # get the user interest vector from the matrix
    user_interest_vector = user_interest_matrix[user_id]
    # now loop throught their interest and find similar interest
    for interest_id, is_interested in enumerate(user_interest_vector):
        if is_interested:
            similar_interest = most_similar_interest_to(interest_id)
            for interest, similarity in similar_interest:
                suggestions[interest] += similarity

    # sort the suggestions by weight (combined similarity)
    suggestions = sorted(suggestions.items(), key=itemgetter(1),
                         reverse=True)

    # determine if we should include their already stated current interest
    # in suggestions
    if include_current_interest:
        return suggestions
    else:
        return [(suggestion, weight) for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]


if __name__ == "__main__":

    # Popularity Recommendation #
    #############################
    print "Popularity Based Recommendations-------------------------------"
    # get the interest ordered by popularity
    popular_interests = count_interests(users_interests)
    # Interest by popularity
    print popular_interests
    
    print "\n"

    # print out user1 recommendations
    print "To user #1 we recommend..."
    print most_popular_new_interests(users_interests[1], users_interests)
    
    print "\n"
   
    # User-Based Collaborative Filtering #
    ######################################
    print "User-Based Similarity Recommendations--------------------------"
    # print user similarity for two sample users
    user_similarities = user_cosine_similarities(users_interests)
    print "User #1 to User #8 similarity is..."
    print user_similarities[0][8]

    # print the most similar users to user 0
    print "Ordered User-Similarity to User 0..."
    print most_similar_users_to(0)

    print "\n"

    # print the user suggestions for user_id[0]
    print "We recommend to user 0 the following..."
    print user_based_suggestions(0)

    print "\n"

    # Item-Based Collaborative Filtering #
    ######################################
    print "Item-Based Similarity Recommendations-------------------------"
    print "The most similar interest to Big Data are..."
    print most_similar_interest_to(0)
    print "\n"
    print "We recommend to user 0 the following..."
    print item_based_recommendations(0)

