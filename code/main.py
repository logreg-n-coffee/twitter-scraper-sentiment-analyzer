"""
@file: main.py
@professor: Dr. Egbert
@author: Rui Hu (Sherman)

*****Python module dependencies*****:
The following are the modules needed for running the program. Please feel free to install them before running.
    1) tweepy - Twitter scraper
    2) pandas (pd) - enhanced dataframe
    3) os (operating system) - for folder and file
    4) datetime - current date and time
    5) textblob (TextBlob) - pretrained machine learning sentiment analyzer
    6) numpy (np) - enhanced scientific calculation
    7) sklearn - 'MaxAbsScaler' from 'sklearn.preprocessing' - used to scale/normalize the data
    8) pingouin (pg) - contains many statistic models (in this program we use this to count Pearson's r)

*****Note*****:
This program let users easily scrape the Tweets of a Twitter user (max Tweets cap: ~3200), and allows sentiment analysis
by using the pretrained machine learning model and the dictionary based model.

If you would like to see the details of each function, you may explore the following two files:
    1) twitter_scraper.py
    2) sentiment_analyzer.py

The following file stores the secrets/keys needed for accessing the Twitter API:
    * settings.py

The following file allows detection of the file path separator according to user's operating system:
    * separator.py

"""

# import everything from twitter_scraper.py and sentiment_analyzer.py
from twitter_scraper import *
from sentiment_analyzer import *

# import time from time
from time import time


# define the main() function
def main(user_screen_name, tweet_count):
    """
    This main function executes all the tasks in this program
    :param: Takes two parameters user_screen_name (string) and tweet_count (integer)
    """
    # load the time count
    initial_start_time = time()

    # run get_tweets() - connect to twitter and process the tweets and writes out the csv file
    print('Getting {} Tweets from @{}. This might take a few seconds.'.format(tweet_count, user_screen_name))
    get_tweets(user_screen_name, tweet_count)
    print('Completed getting Tweets in {} seconds.\n'.format(round(time() - initial_start_time), 3))

    # load a temporary start time
    start_time = time()

    # run clean_tweets() - perform cleaning of the tweets, and save results as csv as a prep for the sentiment analysis
    print('Cleaning Tweets collected from @{}.'.format(user_screen_name))
    clean_tweets(user_screen_name)
    print('Completed cleaning the tweets in {} seconds.\n'.format(round(time() - start_time), 3))

    # reset start time
    start_time = time()

    # analyze sentiment using pretrained ML model and produce a csv file
    print('Analyzing sentiment using the pre-trained machine learning model.')
    analyze_sentiment_pretrained(user_screen_name)
    print('Completed analyzing sentiment using '
          'the pretrained machine learning model in {} seconds.\n'.format(round(time() - start_time), 3))

    # reset start time
    start_time = time()

    # analyze sentiment using dictionary-based model and produce a csv file
    print('Analyzing sentiment using the dictionary-based model.')
    analyze_sentiment_dict_based(user_screen_name)
    print('Completed analyzing sentiment using '
          'the pretrained machine learning model in {} seconds.\n'.format(round(time() - start_time), 3))

    # reset start time
    start_time = time()

    # compare the two models; consolidate the ratings to a single csv file; write out Pearson's r to a csv file
    print('Running correlation analysis on the two models.')
    sentiment_model_correlation(user_screen_name)
    print('Completed correlation analysis on the two models in {} seconds.\n'.format(round(time() - start_time), 3))

    # print out the final summary
    print('All tasks are completed in {} seconds. '
          'Thank you for using me. :) \n'.format(round(time() - initial_start_time), 3))


# run the main()
if __name__ == '__main__':
    # print out welcome message
    print("Hi, welcome to the program. \n"
          "This program allows you to scrape a Twitter user's Tweets and analyze the sentiment. \n")

    # let users choose from the following three options
    choice = input('I have four options for you. Which one would you like? \n'
                   '** Type 1, if you would like to analyze 100 most recent Tweets from @abc \n'
                   '** Type 2, if you would like to analyze 300 most recent Tweets from @abc \n'
                   '** Type 3, if you would like to analyze 3200 most recent Tweets from @abc \n'
                   '** Type 4, if you would like to make your own decision on whose Tweets to analyze.\n')

    # execute the task according to the options
    # Option 1
    if choice == '1':
        print('Thank you for choosing the Option 1. I am working on it!')
        main('abc', 100)

    # Option 2
    elif choice == '2':
        print('Thank you for choosing the Option 2. I am working on it!')
        main('abc', 300)

    # Option 3:
    elif choice == '3':
        print('Thank you for choosing the Option 3. I am working on it!')
        main('abc', 3200)

    # Option 4:
    else:  # all other options
        # welcome message
        print('\n'
              'Oh, wow! So whose Twitter account you are interested in? \n'
              'If there is no input or the name is not correct, I will return the default results from the API user. \n'
              'You may type the screen name of the account followed by the @ sign.')
        screen_name = input('@')

        # count
        print('\n'
              'How many Tweets would you like to collect? \n'
              'Please note that for large numbers, due to internet connections, the results might vary. \n'
              'Please feel free to retry if the results are not what you have expected. \n')
        count = input('Please type an integer from 1 to 3200 (included) here: ')

        # verify the input is an integer and it is within the range of [1, 3200]
        try:
            count_int = int(count)  # if the input is not an integer, ValueError would be caught.
            # the following if-else block would only run if count can be converted to an integer
            if 1 <= count_int <= 3200:
                main(screen_name, count_int)  # run the main()
            else:
                print('Please input an integer from 1 to 3200 (included). Thank you!\n')
        except ValueError:
            print('Sorry. It seems that you did not input a valid integer. \n')
