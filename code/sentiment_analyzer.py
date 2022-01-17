"""
@file: sentiment_analyzer.py
- contains two sentiment analyzers: analyze_sentiment_pretrained() and analyze_sentiment_dict_based()
- analyze_sentiment_pretrained() - pretrained machine learning (ML) model to calculate the sentiment (polarity) rating
- analyze_sentiment_dict_based() - dictionary based model to calculate the sentiment (polarity) rating
@professor: Dr. Egbert
@author: Rui Hu (Sherman)

Note:
This sentiment analyzer is a part of the main.py, but it can run individually.
If you would like to experience the full functionality, please run the main.py
If you would like to run the program, please hit run.
By default, it will:
    1) analyze the sentiment of @abc Tweets using the pretrained ML model; produce 'abc_tweets_pretrained_sentiment.csv'
    2) analyze the sentiment of Tweets using the dictionary-based model; produce 'abc_tweets_dict_based_sentiment.csv'
        produce two frequency dictionaries containing all the positive words and negative words
    3) consolidate the ratings produced by the two models, and produce 'abc_tweets_sentiment_model_ratings.csv'
    4) analyze the correlation of the two models, and write the results to 'abc_tweets_sentiment_model_correlation.csv'
If you would like to make modifications, please change the details in the block of - if __name__ == '__main__'
"""

# import necessary module(s)
from textblob import TextBlob  # pretrained machine learning sentiment analyzer
import pandas as pd  # dataframe
import numpy as np  # enhanced scientific calculation
# from sklearn.preprocessing import MinMaxScaler  # used to scale/normalize the data (not the best solution)
from sklearn.preprocessing import MaxAbsScaler  # used to scale/normalize the data (best solution)
import pingouin as pg  # contains many statistic models (in this program we use this to count Pearson's r)

# import separator() from separator.py
from separator import separator


# pretrained sentiment model
def analyze_sentiment_pretrained(user_screen_name):
    """
    analyze the sentiment using the pretrained model from TextBlob
    :param: user_screen_name
    :return: analyzed pandas dataframe
    """

    # define the path to read csv file from
    csv_file_path = 'output' + separator() + user_screen_name + '_tweets_processed.csv'

    # read the csv into a pandas dataframe named
    # only use the 'text' column, ignore the rest;
    df = pd.read_csv(csv_file_path, usecols=['text'])

    # explicitly state the data type of the dataframe as string to avoid errors
    df = df.astype(str)

    # analyze polarity of the sentiment (neg, pos, neu) - write all the results to a new column named rating
    # if a cell is empty, put the result of the polarity as None
    df['rating_pt'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)  # pt means pretrained

    # write the processed sentiment ratings into 'output' folder
    csv_file_path = 'output' + separator() + user_screen_name + '_tweets_pretrained_sentiment.csv'
    df.to_csv(csv_file_path)

    # returns analyzed dataframe
    return df


# diy dictionary-based sentiment analyzer model
def analyze_sentiment_dict_based(user_screen_name):
    """
    analyze the sentiment using the DIY dictionary based analyzer
    :param user_screen_name
    :return: analyzed pandas dataframe
    """

    # define the path to read csv file from
    csv_file_path = 'output' + separator() + user_screen_name + '_tweets_processed.csv'

    # read the csv into a pandas dataframe named 'df'
    df = pd.read_csv(csv_file_path, usecols=['text'])  # only use the 'text' column, ignore the rest

    # explicitly state the data type of the dataframe as string to avoid errors
    df = df.astype(str)

    # load in sentiment dictionaries (positive words and negative words) - create 2 sets of positive and negative words
    positive_words = set(word.rstrip() for word in open('resources' + separator() + 'positive_words.txt'))
    negative_words = set(word.rstrip() for word in open('resources' + separator() + 'negative_words.txt'))

    # process each tweet in the csv corpus - find positive words and negative words - get the count in each column
    # find positive words for each tweet in the 'text' column
    # split a tweet in each cell in the text column, if it is in the positive_words set, add it to the list
    df['found_positive_words'] = df['text'].apply(lambda x: [x for x in x.split() if x in positive_words])  # make lists
    # the length of the list is the number of words found
    df['count_positive_words'] = df['found_positive_words'].apply(lambda x: len(x))  # length of the list == words found

    # find negative words for each tweet in the 'text' column (similar method)
    df['found_negative_words'] = df['text'].apply(lambda x: [x for x in x.split() if x in negative_words])
    df['count_negative_words'] = df['found_negative_words'].apply(lambda x: len(x))

    # calculate the rating and write it to the column named 'rating'
    # one positive word == 1 point; one negative word == -1 point
    # rating formula = positive score + negative score
    df['rating_raw'] = df['count_positive_words'] - df['count_negative_words']

    # rescale/normalize the rating using min-max normalization method
    # The following block introduces this method: (not encourage)
    # rescaling the dictionary based rating to [-1, 1] is needed - this can be completed using MinMaxScaler()
    # also achievable according to: (2*(x-min(x)))/(max(x)-min(x))-1
    # rescaling_tool = MinMaxScaler(feature_range=[-1, 1])

    # rescale/normalize the rating using maximum absolute value method (Best Solution)
    # dictionary based rating (range is unlimited) does not have the same scale as pretrained ML data [-1, 1]
    # in theory the rating data can expand between negative infinity to positive infinity
    # so we rescale the data to [-1, 1], using formula: value / (the absolute maximum value in the dataset)
    # scikit-learn rescaling tool expects a 2D array - since the data has only a single feature use array.reshape(-1, 1)

    # take out the 'rating_raw' column and convert the data in 'rating_raw' into an numpy array
    rating_raw = np.array(df['rating_raw']).reshape(-1, 1)

    # build an instance of the rescaling_tool using Maximum Absolute Value Method
    rescaling_tool = MaxAbsScaler()

    # fit (load in) the dataset (rating) to be scaled
    rescaling_tool.fit(rating_raw)

    # transform the dataset (rating) into normalized dataset with range of [-1, 1]
    # write out the results to a new column named 'rating'
    df['rating_db'] = rescaling_tool.transform(rating_raw)  # db means pretrained

    # building frequency dictionaries and write out the results

    # process all positive words and make a frequency dictionary 'positive_freq_dict'
    # convert to all positive words into a list from the dataframe
    all_positive_words_raw = df['found_positive_words'].tolist()  # contains empty element
    # remove empty list from list using filter; override 'all_positive_words_raw'
    all_positive_words_raw = list(filter(None, all_positive_words_raw))  # list of lists exist
    # flatten the list from all_negative_words_raw, save the results to all_positive_words
    all_positive_words = list(np.concatenate(all_positive_words_raw).flat)  # np.concatenate(List).flat can

    # initiate a frequency dictionary
    positive_word_freq = dict()
    # For each word in 'all_positive_words', if the word does not exist as a key in the dictionary
    # then add it and set its value to 0. Add 1 to the value.
    for word in all_positive_words:
        positive_word_freq[word] = positive_word_freq.get(word, 0) + 1

    # write out the positive frequency dictionary
    positive_freq_dict_path = 'output' + separator() + user_screen_name + '_tweets_positive_words.csv'  # define path
    positive_freq_dict_file_out = open(positive_freq_dict_path, 'w+')  # open the file for write out
    for i in sorted(positive_word_freq, key=positive_word_freq.get, reverse=True):  # key-value pairs by value
        positive_freq_dict_file_out.write(str(i) + "," + str(positive_word_freq[i]) + "\n")
    positive_freq_dict_file_out.close()  # close the file

    # process all negative words and make a frequency dictionary 'negative_word_freq'
    # convert to all positive words into a list from the dataframe
    all_negative_words_raw = df['found_negative_words'].tolist()  # contains empty element
    # remove empty list from list using filter; override 'all_negative_words_raw'
    all_negative_words_raw = list(filter(None, all_negative_words_raw))  # list of lists exist
    # flatten the list from all_negative_words_raw, save the results to all_negative_words
    all_negative_words = list(np.concatenate(all_negative_words_raw).flat)  # np.concatenate(List).flat can flat lists

    # initiate a frequency dictionary
    negative_word_freq = dict()
    # For each word in 'all_negative_words', if the word does not exist as a key in the dictionary
    # then add it and set its value to 0. Add 1 to the value.
    for word in all_negative_words:
        negative_word_freq[word] = negative_word_freq.get(word, 0) + 1

    # write out the negative frequency dictionary
    negative_freq_dict_path = 'output' + separator() + user_screen_name + '_tweets_negative_words.csv'  # define path
    negative_freq_dict_file_out = open(negative_freq_dict_path, 'w+')  # open the file for write out
    for i in sorted(negative_word_freq, key=negative_word_freq.get, reverse=True):  # key-value pairs by value
        negative_freq_dict_file_out.write(str(i) + "," + str(negative_word_freq[i]) + "\n")
    negative_freq_dict_file_out.close()  # close the file

    # write out the tweets_dict_based_sentiment csv to the folder
    # for beauty use comma separated strings in the output - override the cell in column named 'found_positive_words'
    df['found_positive_words'] = df['found_positive_words'].apply(lambda x: ', '.join(x))  # comma separated strings
    # for beauty use comma separated strings in the output - override the cell in column named 'found_negative_words'
    df['found_negative_words'] = df['found_negative_words'].apply(lambda x: ', '.join(x))  # comma separated strings

    # write the processed sentiment ratings into 'output' folder
    csv_file_path = 'output' + separator() + user_screen_name + '_tweets_dict_based_sentiment.csv'
    df.to_csv(csv_file_path)

    # return analyzed dataframe
    return df


# sentiment model comparison tool -
# write out two ratings datasets to a new csv file; create a new csv file to store the correlation data
def sentiment_model_correlation(user_screen_name):
    """
    compare two models by calculating the correlation between the two data set
    :param user_screen_name
    :return: None
    """
    # specify the paths of csv files
    pt_csv_file_path = 'output' + separator() + user_screen_name + '_tweets_pretrained_sentiment.csv'  # pt - pretrained
    db_csv_file_path = 'output' + separator() + user_screen_name + '_tweets_dict_based_sentiment.csv'  # db - dict-based

    # load in two csv files
    pt_df = pd.read_csv(pt_csv_file_path)
    db_df = pd.read_csv(db_csv_file_path)

    # write the two ratings  into 'output' folder (to facilitate analysis on other software)
    df = pd.DataFrame()  # create a new dataframe named 'df'
    df['rating_pt'] = pt_df['rating_pt']  # add rating_pt from pt_df to a newly created df named df
    df['rating_db'] = db_df['rating_db']  # add rating_db from db_df to a newly created df named df
    df_csv_file_path = 'output' + separator() + user_screen_name + '_tweets_sentiment_model_ratings.csv'
    df.to_csv(df_csv_file_path)

    # use built in statistic model to calculate Pearson's r (correlation) and write the results to a csv file
    # specify the write-out directory
    correlation_csv_file_path = 'output' + separator() + user_screen_name + '_tweets_sentiment_model_correlation.csv'
    # pg.corr will generate a pandas dataframe so it is possible to write out with built in .to_csv
    pg.corr(pt_df['rating_pt'], db_df['rating_db']).to_csv(correlation_csv_file_path)

    # return
    return None


"""
Please make changes to the following block to diy your test of this program
"""
if __name__ == '__main__':
    analyze_sentiment_pretrained('abc')  # analyze the sentiment of @abc Tweets using the pretrained ML model
    analyze_sentiment_dict_based('abc')  # analyze the sentiment of Tweets using the dictionary-based model
    sentiment_model_correlation('abc')  # consolidate the ratings produced; analyze the correlation of the two models
    print('Tasks completed. Please check the files in the output folder.')
