"""
@file: twitter_scraper.py
- definition of the Twitter scrapper based on Tweepy
- scrape the tweets of anyone with their user_id and write the tweets into a csv file
@professor: Dr. Egbert
@author: Rui Hu 

Note:
This twitter scraper is a part of the main.py, but it can run individually.
If you would like to experience the full functionality, please run the main.py
If you would like to run the program, please hit run.
By default, it will:
    1) get 100 tweets from @abc, and write out the csv file named 'abc_tweets.csv'
    2) clean the tweets obtained from @abc, and write out the csv file named 'abc_tweets_processed.csv'
If you would like to make modifications, please change the details in the block of - if __name__ == '__main__'
"""

# import necessary module
import tweepy  # Twitter scraper
import pandas as pd  # dataframe
import os  # operating system (for folder and file)
from datetime import datetime  # current date and time
# import re

# import settings from the settings.py file
import settings

# import separator() from separator.py file
from separator import separator

# load the settings for Tweepy from the setting file
consumer_key = settings.TWEEPY_SETTINGS['consumer_key']
consumer_secret = settings.TWEEPY_SETTINGS['consumer_secret']
key = settings.TWEEPY_SETTINGS['key']
secret = settings.TWEEPY_SETTINGS['secret']

# initiate a dictionary cleaned_tweets to store cleaned tweets
cleaned_tweets: dict = {
    'mined_at': [],
    'created_at': [],
    'tweet_id': [],
    'name': [],
    'screen_name': [],
    'lang': [],
    'favorite_count': [],
    'retweet_count': [],
    # 'mentions': [],
    # 'hashtags': [],
    # 'location': [],
    'text': [],
}


def get_tweets(user_screen_name, tweet_count):
    """
    connect to twitter and process the tweets and writes out the csv file
    :param: take user_id and tweet_count as arguments (max==~3200 as confined by Twitter Corporation)
    :return: pandas dataframe
    """
    # initiate a Tweepy instance

    # authentication (OAuth 1a Authentication) using Twitter developer account

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(key, secret)

    # initiate api
    api = tweepy.API(auth)

    # retrieve Tweets from the user's timeline - raw data format: json (multidimensional)
    # https://developer.twitter.com/en/docs/twitter-api/v1/tweets/timelines/api-reference/get-statuses-user_timeline

    # the ordinary way - api.user_timeline(id=user_screen_name, count=tweet_count) only obtain 200 tweets (statuses)
    # tweets = api.user_timeline(id=user_screen_name, count=tweet_count)  # 'tweets' is a tweepy <status> object

    # Use tweepy.Cursor to catch more tweets (subject to the maximum tweet count of a user and Twitter limit of ~3200)
    # introduce tweepy Cursor to perform pagination
    # https://docs.tweepy.org/en/v3.10.0/cursor_tutorial.html

    # process the tweets and make a dictionary (2-dimensional) - flattening the json source
    for tweet in tweepy.Cursor(api.user_timeline, id=user_screen_name).items(tweet_count):
        # add the following pair to the dictionary
        cleaned_tweets['mined_at'].append(datetime.now())
        cleaned_tweets['created_at'].append(tweet.created_at)
        cleaned_tweets['tweet_id'].append(tweet.id_str)  # id_str returns numbers in string, id also works
        cleaned_tweets['name'].append(tweet.user.name)
        cleaned_tweets['screen_name'].append(tweet.user.screen_name)
        cleaned_tweets['lang'].append(tweet.lang)
        cleaned_tweets['favorite_count'].append(tweet.favorite_count)
        cleaned_tweets['retweet_count'].append(tweet.retweet_count)
        # cleaned_tweets['mentions'].append(tweet.entities['user_mentions']['screen_name']
        # cleaned_tweets['hashtags'].append(tweet.entities['hashtags']['text'])
        # cleaned_tweets['location'].append(tweet.place)
        cleaned_tweets['text'].append(tweet.text)

        # debugging
        # print(cleaned_tweets)

    # convert the cleaned_tweets into pandas dataframe
    cleaned_tweets_dataframe = pd.DataFrame(cleaned_tweets)

    # calculate the average length of the tweets (max == 140 characters) and add a new column to the dataframe
    cleaned_tweets_dataframe['text_length'] = cleaned_tweets_dataframe['text'].str.len()

    # write the Tweets of the user into a csv file into 'output' folder
    if not os.path.exists('output'):
        os.makedirs('output')

    csv_file_path = 'output' + separator() + user_screen_name + '_tweets.csv'
    cleaned_tweets_dataframe.to_csv(csv_file_path)

    # return the dataframe
    return cleaned_tweets_dataframe


# Further clean Tweet text as there are some noise
def clean_tweets(user_screen_name):
    """
    perform cleaning of the tweets, and save results as csv - as prep for sentiment analysis
    :param: user_screen_name
    :return: tweets_text dataframe
    """

    # define the path to read csv file from
    csv_file_path = 'output' + separator() + user_screen_name + '_tweets.csv'

    # create a set of stopwords
    # from: https://github.com/stopwords-iso/stopwords-en - also built in nltk module
    stopwords = set(word.rstrip() for word in open('resources' + separator() + 'stopwords.txt'))

    # read the csv from the path and load the dataframe as df
    df = pd.read_csv(csv_file_path, usecols=['text'])  # only use the 'text' column, ignore the rest

    # explicitly state the data type of the dataframe as string to avoid errors
    df = df.astype(str)

    # clean the text in tweets by removing links and special characters using regex (override the original data)
    # pandas allows batch replace with regex with its built in method replace()
    # can also be written as:
    # df['text'] = df['text'].apply(
    #     lambda x: ' '.join(re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', x).split()))
    df['text'] = df['text'].replace(to_replace=r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)',
                                    value=' ', regex=True)

    # remove the stop words in the text (override the original data)
    # pandas allows batch function/method apply with built-in method apply()
    # lambda - non-declared function returns the tweet without the stopwords
    df['text'] = df['text'].apply(lambda x: ' '.join(x for x in x.split() if x not in stopwords))

    # lowercase the words in the text (override the original data)
    df['text'] = df['text'].apply(lambda x: ' '.join(x.lower() for x in x.split()))

    # write the processed Tweets of the user into a csv file into 'output' folder
    csv_file_path = 'output' + separator() + user_screen_name + '_tweets_processed.csv'
    df.to_csv(csv_file_path)


"""
Please make changes to the following block to diy your test of this program
"""
if __name__ == '__main__':
    get_tweets('abc', 100)  # get 100 tweets from @abc, and write out the csv file named 'abc_tweets.csv'
    clean_tweets('abc')  # clean the tweets obtained from @abc, and write out the file named 'abc_tweets_processed.csv'
    print('Tasks completed. Please check the files in the output folder.')
