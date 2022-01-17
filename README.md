
# Building a Twitter Scraper and a Prototype Dictionary-based Sentiment Analyzer

 - Author: Rui Hu 
 - Course: Programming for text analysis 
 - Professor: Dr. Jesse Egbert 
 - Date: Tuesday, April 27, 2021

**Note: Tweepy needs Twitter developer account for assessing Twitter API data. If you would like to run the code on your computer, you need a set of Twitter API keys.**

## Structure of this repository

### 1. Code

#### Py files:

      1.  Main Program - main.py
      2.  Twitter Scraper -- twitter_scraper.py
      3.  Sentiment Analyzer -- sentiment_analyzer.py
      4.  Settings for the Twitter API -- settings.py
      5.  Automatic Separator Detection (for Windows & Mac Users) --
          separator.py

#### Resources:

      1.  Positive word list
      2.  Negative word list
      3.  Stop word list
      4.  Citations for the above three lists

### 2. Data
This includes ABC News Tweets data generated by the program. I used the data for the documentation.

### 3. Presentation
This includes my full presentation slides for Tuesday, April 27, 2021.

### 4. Document
This saves the document of this project.

# Dependencies

  1.  tweepy - Twitter scraper
  2.  pandas (pd) - enhanced dataframe
  3.  os (operating system) - for folder and file
  4.  datetime - current date and time
  5.  textblob (TextBlob) - pretrained machine learning sentiment analyzer
  6.  numpy (np) - enhanced scientific calculation
  7.  sklearn - \'MaxAbsScaler\' from \'sklearn.preprocessing\' - used to
      scale/normalize the data
  8.  pingouin (pg) - contains many statistic models (in this program we
      use this to calculate Pearson\'s r)