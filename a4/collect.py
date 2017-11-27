from collections import Counter
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings("ignore")
import collections

consumer_key = 'ziK1kNJDiMUFz9ig8IbYwbQdo'
consumer_secret = 'cLe4ITEAxcCHmkfgoeqMgM5tpC93WP5UckJ7D2DjpXxgti9M5X'
access_token = '1710128701-WKcXN4vNYZyB9mrUOFCd62X6zKgocfdtaaaU5w5'
access_token_secret = '588ItsTG6vdyWqFqOnzpoXwuKQk6YMllDlEB4IPJ8ojZ8'

def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def read_screen_names(filename):
    """
    Read a text file containing Twitter screen_names, one per line.

    Params:
        filename....Name of the file to read.
    Returns:
        A list of strings, one per screen_name, in the order they are listed
        in the file.

    Here's a doctest to confirm your implementation is correct.
    >>> read_screen_names('candidates.txt')
    ['DrJillStein', 'GovGaryJohnson', 'HillaryClinton', 'realDonaldTrump']
    """
    ###TODO
    screen_name = []
    file = open('candidate.txt','r').read().split('\n')
    for line in file:
        if line.strip():
            screen_name.append(line)   
    return screen_name

def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

def get_users(twitter, screen_names):
    """Retrieve the Twitter user objects for each screen_name.
    Params:
        twitter........The TwitterAPI object.
        screen_names...A list of strings, one per screen_name
    Returns:
        A list of dicts, one per user, containing all the user information
        (e.g., screen_name, id, location, etc)

    See the API documentation here: https://dev.twitter.com/rest/reference/get/users/lookup

    In this example, I test retrieving two users: twitterapi and twitter.

    >>> twitter = get_twitter()
    >>> users = get_users(twitter, ['twitterapi', 'twitter'])
    >>> [u['id'] for u in users]
    [6253282, 783214]
    """
    ###TODO
    
    request = robust_request(twitter,'users/lookup',{'screen_name':screen_names})
    users = [r for r in request]
    return users

def get_friends(twitter, screen_name):
    """ Return a list of Twitter IDs for users that this person follows, up to 5000.
    See https://dev.twitter.com/rest/reference/get/friends/ids

    Note, because of rate limits, it's best to test this method for one candidate before trying
    on all candidates.

    Args:
        twitter.......The TwitterAPI object
        screen_name... a string of a Twitter screen name
    Returns:
        A list of ints, one per friend ID, sorted in ascending order.

    Note: If a user follows more than 5000 accounts, we will limit ourselves to
    the first 5000 accounts returned.

    In this test case, I return the first 5 accounts that I follow.
    >>> twitter = get_twitter()
    >>> get_friends(twitter, 'aronwc')[:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    ###TODO
    request = robust_request(twitter,'friends/ids',{'screen_name': screen_name ,'count': 5000})
    friends = [r for r in request]
    friends.sort()
    return friends

def add_all_friends(twitter, users):
    """ Get the list of accounts each user follows.
    I.e., call the get_friends method for all 4 candidates.

    Store the result in each user's dict using a new key called 'friends'.

    Args:
        twitter...The TwitterAPI object.
        users.....The list of user dicts.
    Returns:
        Nothing

    >>> twitter = get_twitter()
    >>> users = [{'screen_name': 'aronwc'}]
    >>> add_all_friends(twitter, users)
    >>> users[0]['friends'][:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    ###TODO
    for u in users:
        u['friends'] = get_friends(twitter,u['screen_name'])
        
        
def friends_lookup(twitter, users, friends):
    for u in friends:
        request = robust_request(twitter,'users/lookup',{'user_id':u})
        users.extend([r for r in request])
        
def friend_of_friend(twitter,users):
    for i in range(1,len(users)):
        users[i]['friends'] = get_friends(twitter,users[i]['screen_name'])
        
def num_collect(users):
    edges ={}
    file = open('follower.txt','w')
    for i in range(len(users)):
        edges[users[i]['screen_name']] = users[i]['friends']
        for f in users[i]['friends']:        
            file.write("{}\t{}\n".format(users[i]['screen_name'], f))        
    file.truncate(file.tell()-2)
    file.close()
        
def main():

	twitter = get_twitter()
	screen_names = read_screen_names('candidate.txt')
	print('Established Twitter connection.')
	print('Read screen names: %s' % screen_names)
	users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
	add_all_friends(twitter, users)
	friends = users[0]['friends']
	friends_lookup(twitter, users, friends)
	friend_of_friend(twitter,users)
	num_collect(users)
	print('Data Collected check follower.txt file.')
	print('Run the next script cluster.py.')
    
if __name__ == '__main__':
    main()
