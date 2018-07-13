import os
import numpy as np
import tensorflow as tf

from feed import feed_stories, decode_to_string, decode_source, NUM_SOURCES, NUM_CLASSES, ZERO_CLASS
from model import model, model_scope

def write_to_file( filename, text, overwrite = False ):
    with open( filename, 'w' if overwrite else 'a', encoding = 'utf-8' ) as file:
        file.write( '\n'.join( text ) )

def generate( source, initial_char, batch_size, training = False ):
    def choose_char( character_logits ):
        '''
        Choose a character from the logits representing the weighted distribution
        '''
        chosen_output = tf.multinomial( character_logits, 1, output_dtype = tf.int32 )
        return tf.squeeze( chosen_output, [-1] )

    def body( i, state, char_array ): 
        # Pass in previous input get next output
        model_input = tf.one_hot( char_array.read( i - 1 ), NUM_CLASSES )
        model_output, next_state = model( state, model_input, training, create_state = False )
        # Choose a character from the logits representing the weighted distribution
        chosen_output = choose_char( model_output )
        # Place character into char array for next go.
        char_array = char_array.write( i, chosen_output )
        return i + 1, next_state, char_array

    def cond( i, _state, char_array ):
        return tf.logical_not( tf.logical_or(
            tf.reduce_all( tf.equal( char_array.read( i - 1 ), ZERO_CLASS ) ),
            tf.equal( i, 128 )
        ) )

    # Create an array to store extrapolated segments
    char_array = tf.TensorArray( tf.int32, size = 1, dynamic_size = True, element_shape = [batch_size], clear_after_read = False )

    # Produce our first segment and place it into the char array
    first_output, initial_state = model( source, tf.one_hot( initial_char, NUM_CLASSES ), training = training )
    char_array = char_array.write( 0, choose_char( first_output ) )

    # Produce the rest of the segments
    _, _, char_array = tf.while_loop( cond, body, [tf.constant( 1 ), initial_state, char_array], back_prop = training )

    char_array = char_array.stack() # Stack them up into a single tensor, [chars, batch]
    char_array = tf.transpose( char_array, [1, 0] ) # [batch, chars]

    return char_array

def train():
    BATCH_SIZE = 32

    global_step = tf.train.create_global_step()
    global_step_inc = global_step.assign_add( 1 )

    sources, titles = feed_stories( BATCH_SIZE )

    with tf.variable_scope( 'model' ):
        source_vectors = tf.one_hot( sources, NUM_SOURCES )
        title_vectors = tf.one_hot( titles[:, :-1], NUM_CLASSES )
        predicted_logits, _ = model( source_vectors, title_vectors )

        cost = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( logits = predicted_logits, labels = titles[:, 1:] ) )

        # Have some output stuff
        predicted_classes = tf.concat( (titles[:, 0:1], tf.argmax( predicted_logits, -1, output_type = tf.int32 )), 1 )
        generated_classes = tf.concat( (titles[:, 0:1], generate( source_vectors, titles[:, 0], BATCH_SIZE, training = False )), 1 )

    with tf.variable_scope( 'train' ):
        learning_rate = tf.Variable( 1e-4, trainable = False )
        lr_adjust = learning_rate.assign( learning_rate * 0.9 )
        trainer = tf.train.AdamOptimizer( learning_rate ).minimize( cost )

    with tf.Session( config = tf.ConfigProto( gpu_options = tf.GPUOptions( per_process_gpu_memory_fraction = 0.75 ) ) ) as session:
        session.run( tf.global_variables_initializer() )

        saver = tf.train.Saver()
        saver.export_meta_graph( './save/model.meta' )
        saver.restore( session, './save/model' )

        write_to_file( './data/inputs.txt', decode_to_string( session.run( titles ) ), overwrite = True )

        try:
            while True:
                i = session.run( global_step_inc )
                _cost, _predictions, _ = session.run( [cost, predicted_classes, trainer] )
                print( '{:05d}: {:6.4f}, {:}'.format( i, _cost, decode_to_string( _predictions )[0] ) )

                if i % 200 == 0:
                    write_to_file( './data/generated.txt', [str(i), *decode_to_string( session.run( generated_classes ) ), '\n'] )
                if i % 500 == 0 and i <= 10000:
                    session.run( lr_adjust )
                if i % 1000 == 0:
                    saver.save( session, './save/model' )
        except KeyboardInterrupt:
            print( 'saving' )
        
        saver.save( session, './save/model' )

if __name__ == '__main__':
    train()
