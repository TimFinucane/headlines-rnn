# TODO: Attrib news api

import numpy as np
import tensorflow as tf

from source import NEWS_SOURCES

NEWS_NAMES = [
    'ABC News',
    'BBC News',
    'Al Jazeera English',
    'Ars Technica',
    'CNN',
    'Fox News',
    'NBC News',
    'Breitbart News',
    'Daily Mail',
    'Reuters',
    'The Irish Times',
    'The Guardian UK',
    'The Guardian (AU)',
    'NBC News'
]
NUM_SOURCES = len(NEWS_NAMES)

CHAR_CLASSES = [
    " ", "!", '"', "#", "$", "%", "&", "'", "(", ")", "*",
    "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5",
    "6", "7", "8", "9", ":", ";", "?", "A", "B", "C", "D",
    "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "[", "]", "_", "`", "a", "b", "c", "d", "e", "f", "g",
    "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
    "s", "t", "u", "v", "w", "x", "y", "z", "¡", "£", "®",
    "°", "´", "½", "¾", "¿", "Á", "É", "Ó", "Ú", "à", "á",
    "â", "ã", "ä", "æ", "ç", "è", "é", "í", "ï", "ñ", "ò",
    "ó", "ö", "ø", "ú", "ü", "š", "˜", "–", "—", "‘", "’",
    "“", "”", "…", "€", "™", "\0"]
NUM_CLASSES = len(CHAR_CLASSES)
ZERO_CLASS = CHAR_CLASSES.index( "\0" )

# Encoding and decoding strings
def decode_source( sources ):
    '''
    Converts a source id into a source name
    '''
    return np.take( NEWS_NAMES, sources )

def encode_source( source ):
    '''
    Converts a source name into a source id
    '''
    return NEWS_NAMES.index( source.decode( 'ascii' ) )

def encode_string( string ):
    '''
    Converts a headline string into a series of classes
    '''
    return np.array( [CHAR_CLASSES.index( c ) for c in string.decode( 'utf-8' )], np.int32 )

def decode_to_string( encodings ):
    '''
    Converts encoded classes into a string
    '''
    return [''.join( [CHAR_CLASSES[i] for i in str_indices if i != ZERO_CLASS] ) for str_indices in encodings]

# Feeding into the training process
def feed_stories( batch_size ):
    dataset = tf.data.TextLineDataset( './data/store.txt' ).map(
        lambda line: tf.split( tf.string_split( [line], '|', skip_empty = False ).values, num_or_size_splits = 3, axis = 0 )
    ).map(
        lambda source, title, desc: tf.py_func(
            lambda source, title, desc: (encode_source( source[0] ), encode_string( title[0] )),
            [source, title, desc],
            [tf.int32, tf.int32]
        )
    )

    dataset = dataset.repeat().padded_batch( batch_size, ([], [None]), (0, ZERO_CLASS) )
    it = dataset.make_initializable_iterator()

    return it.initializer, it.get_next()