'''
All about sourcing the data store.
Thanks to NewsApi.org for allowing me to retrieve news headlines through a free account.
'''
from newsapi import NewsApiClient
import numpy as np

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
    
    with open( './data/store.txt', 'a' if append else 'x', encoding = 'utf-8' ) as store:
        for i in range( 1, pages ):
            articles = news.get_everything( 
                sources = ','.join( NEWS_SOURCES ),
                to = '2018-04-02',
                page_size = 100,
                page = i
            )
            
            print( 'Read page {:}'.format( i ) )
            for article in articles['articles']:
                try:
                    text = []
                    for part in [article['source']['name'], article['title'], article['description']]:
                        part.replace( '\n', '. ' )
                        part.replace( ' | ', ' ' )
                        part.replace( '|', ' ' )
                        text.append( part )

                    store.write( '|'.join( text ) + '\n' )
                except Exception:
                    continue

if __name__ == '__main__':
    store_stories( append = True, pages = 99 )