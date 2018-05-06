import tensorflow as tf

with tf.variable_scope( 'char_lstm' ) as model_scope:
    pass

def model( source, char_inputs, training = True, create_state = True ):
    NUM_LAYERS = 3
    HIDDEN_UNITS = 1024

    if len( char_inputs.shape ) == 2:
        single_input = True
    else:
        single_input = False

    with tf.variable_scope( model_scope, reuse = tf.AUTO_REUSE ):
        if create_state:
            state = tf.layers.dense( source, NUM_LAYERS * HIDDEN_UNITS )
            state = tf.reshape( state, [-1, NUM_LAYERS, HIDDEN_UNITS] )
            state = tf.transpose( state, [1, 0, 2] )
            state = state + tf.random_normal( tf.shape( state ), stddev = 0.2 )
        else:
            state = source

        if single_input:
            char_inputs = tf.expand_dims( char_inputs, [1] )

        lstm_inputs = tf.layers.conv1d( char_inputs, HIDDEN_UNITS, 1, name = 'embedding_in' )
        lstm_inputs = tf.transpose( lstm_inputs, [1, 0, 2] )

        lstm = tf.contrib.cudnn_rnn.CudnnGRU( NUM_LAYERS, HIDDEN_UNITS )
        lstm_outputs, output_state = lstm( lstm_inputs, (state,), training = training )

        lstm_outputs = tf.transpose( lstm_outputs, [1, 0, 2] ) # [batch, chars, one_hot]
        char_outputs = tf.layers.conv1d( lstm_outputs, char_inputs.shape[-1], 1, name = 'embedding_out' )

        if single_input:
            char_outputs = tf.squeeze( char_outputs, [1] )

        return char_outputs, output_state[0]
