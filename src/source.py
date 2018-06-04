'''
All about sourcing the data store.
Thanks to NewsApi.org for allowing me to retrieve news headlines through a free account.
'''
from newsapi import NewsApiClient
import numpy as np

import csv

NEWS_SOURCES = [
    'abc-news',
    'bbc-news',
    'al-jazeera-english',
    'ars-technica',
    'cnn',
    'fox-news',
    'nbc-news',
    'breitbart-news',
    'daily-mail',
    'reuters',
    'the-irish-times',
    'the-guardian-uk',
    'nbc-news'
]

# Storing news headlines into a text file
def store_stories( append = True, pages: int = 5 ):
    with open( './data/newsapi.org.key' ) as file:
        news = NewsApiClient( api_key = file.read() )
    
    with open( './data/headlines.csv', 'a' if append else 'w', encoding = 'utf-8', newline = '' ) as headlines_file:
        headlines_writer = csv.writer( headlines_file )
        for i in range( 1, pages ):
            articles = news.get_everything( 
                sources = ','.join( NEWS_SOURCES ),
                to = '2018-05-30',
                page_size = 100,
                page = i
            )
            
            # possible regex for removing other languages (non-european) \n[^\n]*[^\x00-\x7F]{5}[^\n]*(?=\n)
            print( 'Read page {:}'.format( i ) )
            for article in articles['articles']:
                try:
                    headlines_writer.writerow( [article['source']['name'], article['title'], article['description']] )
                except Exception:
                    continue

if __name__ == '__main__':
    store_stories( append = True, pages = 100 )