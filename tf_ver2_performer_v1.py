import tensorflow as tf
from tensorflow.keras.layers import Embedding, LayerNormalization

# Multi-Head Attention Layer. #
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(
        self, d_model, n_heads, r_kernel, 
        seed=1234, name="multi_head_attn_layer"):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        d_depth = int(d_model / n_heads)
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_depth = d_depth
        self.r_kernel = r_kernel
        self.ker_shape = (n_heads, d_depth, r_kernel)
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.wc = tf.keras.layers.Dense(d_model)
        self.wg = tf.keras.initializers.Orthogonal(
            gain=1.0, seed=seed)(shape=self.ker_shape)
    
    def split_heads(self, x):
        # Input is (batch_size, seq_len, d_model). #
        # Output is (batch_size, num_heads, seq_len, depth). #
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        output_shp = (batch_size, seq_length, 
                      self.n_heads, self.d_depth)
        
        x = tf.reshape(x, output_shp)
        return tf.transpose(x, [0, 2, 1, 3])
    
    def combine_heads(self, x):
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[2]
        output_shp = (
            batch_size, seq_length, self.d_model)
        
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, output_shp)
    
    # Softmax Approximation. #
    def sm_approx(self, x):
        rsqr_d = tf.math.rsqrt(float(self.d_depth))
        norm_x = tf.reduce_sum(
            tf.square(x), axis=-1, keepdims=True)
        
        h_x = tf.exp(-0.5 * norm_x)
        phi_x = tf.multiply(
            rsqr_d, tf.exp(tf.matmul(x, self.wg)))
        x_prime = h_x * phi_x
        return x_prime
    
    def call(
        self, v, k, q, k_prev=None, 
        kv_prev=None, eps=1.0e-6):
        norm_d = tf.math.rsqrt(
            tf.math.sqrt(float(self.d_depth)))
        
        if k_prev is None:
            k_prev = 0.0
        else:
            k_prev = tf.expand_dims(k_prev, axis=2)
        
        if kv_prev is None:
            kv_prev = 0.0
        else:
            kv_prev = tf.expand_dims(kv_prev, axis=2)
        
        q = self.split_heads(self.wq(q) * norm_d)
        k = self.split_heads(self.wk(k) * norm_d)
        v = tf.expand_dims(
            self.split_heads(self.wv(v)), axis=3)
        
        q_prime = self.sm_approx(q)
        k_prime = self.sm_approx(k)
        q_prime = tf.expand_dims(q_prime, axis=3)
        k_prime = tf.expand_dims(k_prime, axis=4)
        
        kv_prime  = tf.matmul(k_prime, v)
        k_prefix  = tf.add(
            k_prev, tf.math.cumsum(k_prime, axis=2))
        kv_prefix = tf.add(
            kv_prev, tf.math.cumsum(kv_prime, axis=2))
        qk_prefix = tf.squeeze(
            tf.matmul(q_prime, k_prefix), axis=4)
        
        attn_unnorm = tf.matmul(q_prime, kv_prefix)
        attn_unnorm = tf.squeeze(attn_unnorm, axis=3)
        
        # Normalise to get softmax approximation. #
        attn_outputs = tf.divide(
            attn_unnorm, eps + qk_prefix)
        attn_outputs = self.wc(
            self.combine_heads(attn_outputs))
        return k_prefix, kv_prefix, attn_outputs
        
class FFWNetwork(tf.keras.layers.Layer):
    def __init__(self, d_ffwd, d_model):
        super(FFWNetwork, self).__init__()
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        
        self.ffwd_1 = tf.keras.layers.Dense(
            d_ffwd, activation="relu")
        self.ffwd_2 = tf.keras.layers.Dense(d_model)
    
    def call(self, x):
        return self.ffwd_2(self.ffwd_1(x))

# GPT Decoder Layer. #
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self, d_model, n_heads, d_ffwd, 
        r_kernel, seed=1234, rate1=0.1, 
        rate2=0.1, name="decoder_layer"):
        super(DecoderLayer, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.ffwd_self = FFWNetwork(d_ffwd, d_model)
        self.attn_self = MultiHeadAttention(
            d_model, n_heads, r_kernel, seed=seed, name=name)
        
        self.lnorm_1 = LayerNormalization(epsilon=1.0e-6)
        self.lnorm_2 = LayerNormalization(epsilon=1.0e-6)
        self.dropout_1 = tf.keras.layers.Dropout(rate1)
        self.dropout_2 = tf.keras.layers.Dropout(rate2)
    
    def call(
        self, x_enc, x_pos, k_prev=None, 
        kv_prev=None, training=True):
        x_embed = x_enc + x_pos
        attn_tuple = self.attn_self(
            x_embed, x_embed, x_embed, 
            k_prev=k_prev, kv_prev=kv_prev)
        
        k_prefix  = attn_tuple[0]
        kv_prefix = attn_tuple[1]
        attn_self_output = attn_tuple[2]
        
        # Apply Normalisation followed by adding. #
        attn_self_output = self.dropout_1(
            attn_self_output, training=training)
        attn_self_output = tf.add(
            x_embed, self.lnorm_1(attn_self_output))
        
        ffwd_self_output = self.lnorm_2(
            self.ffwd_self(attn_self_output))
        ffwd_self_output = tf.add(
            attn_self_output, ffwd_self_output)
        ffwd_self_output = self.dropout_2(
            ffwd_self_output, training=training)
        return k_prefix, kv_prefix, ffwd_self_output

class Decoder(tf.keras.layers.Layer):
    def __init__(
        self, n_layers, d_model, n_heads, 
        d_ffwd, vocab_size, max_seq_length, 
        seed=1234, rate1=0.1, rate2=0.1, r_kernel=None):
        super(Decoder, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.d_rsqrt = tf.math.sqrt(
            tf.cast(d_model, tf.float32))
        self.vocab_size = vocab_size
        
        if r_kernel is None:
            self.r_kernel = int(d_model / n_heads)
        else:
            self.r_kernel = r_kernel
        
        # Embedding layers. #
        tmp_pos_embed = []
        for n_layer in range(n_layers):
            tmp_pos_embed.append(
                Embedding(max_seq_length, d_model))
        
        self.pos_embed = tmp_pos_embed
        self.dec_embed = Embedding(vocab_size, d_model)
        del tmp_pos_embed
        
        # Decoder Layers. #
        tmp_dec_layers = []
        for n_layer in range(n_layers):
            tmp_dec_layers.append(DecoderLayer(
                d_model, n_heads, 
                d_ffwd, r_kernel, 
                seed=seed, rate1=rate1, rate2=rate2, 
                name="decoder_layer_" + str(n_layer+1)))
        
        self.dec_layers  = tmp_dec_layers
        self.emb_dropout = tf.keras.layers.Dropout(rate1)
        del tmp_dec_layers
    
    def call(
        self, x, l_st=0, k_prev=None, 
        kv_prev=None, training=True):
        seq_length = tf.shape(x)[1]
        
        x_pos_index = tf.expand_dims(
            l_st + tf.range(seq_length), axis=0)
        x_tok_embed = self.dec_embed(x)
        x_tok_embed = self.emb_dropout(
            x_tok_embed * self.d_rsqrt, training=training)
        
        if k_prev is None:
            k_prev = [None for _ in range(self.n_layers)]
        if kv_prev is None:
            kv_prev = [None for _ in range(self.n_layers)]
        
        k_curr_prefix  = []
        kv_curr_prefix = []
        
        layer_input = x_tok_embed
        for m in range(self.n_layers):
            x_pos_embed = self.pos_embed[m](x_pos_index)
            x_pos_embed = self.emb_dropout(
                x_pos_embed * self.d_rsqrt, training=training)
            
            output_tuple = self.dec_layers[m](
                layer_input, x_pos_embed, k_prev=k_prev[m], 
                kv_prev=kv_prev[m], training=training)
            
            k_prefix  = output_tuple[0][:, :, -1, :, :]
            kv_prefix = output_tuple[1][:, :, -1, :, :]
            
            layer_input = output_tuple[2]
            k_curr_prefix.append(
                tf.expand_dims(k_prefix, axis=0))
            kv_curr_prefix.append(
                tf.expand_dims(kv_prefix, axis=0))
        
        k_curr_prefix  = tf.concat(k_curr_prefix, axis=0)
        kv_curr_prefix = tf.concat(kv_curr_prefix, axis=0)
        return k_curr_prefix, kv_curr_prefix, output_tuple[2]

class GPTPerformer(tf.keras.Model):
    def __init__(
        self, n_layers, n_heads, d_model, 
        d_ffwd, vocab_size, max_seq_length, 
        seed=1234, rate1=0.1, rate2=0.1, r_kernel=None):
        super(GPTPerformer, self).__init__()
        assert d_model % n_heads == 0
        
        self.rate1 = rate1
        self.rate2 = rate2
        self.n_heads  = n_heads
        self.n_layers = n_layers
        
        self.d_ffwd  = d_ffwd
        self.d_model = d_model
        self.seq_len = max_seq_length
        self.vocab_size = vocab_size
        
        if r_kernel is None:
            self.r_kernel = int(d_model / n_heads)
        else:
            self.r_kernel = r_kernel
        
        # Output projection. #
        self.gpt_model = Decoder(
            n_layers, d_model, n_heads, d_ffwd, 
            vocab_size, max_seq_length, seed=seed, 
            rate1=rate1, rate2=rate2, r_kernel=r_kernel)
        self.p_decoder = tf.keras.layers.Dense(vocab_size)
    
    def call(
        self, x, l_st=0, k_prev=None, 
        kv_prev=None, training=True):
        output_tuple = self.gpt_model(
            x, l_st=l_st, k_prev=k_prev, 
            kv_prev=kv_prev, training=training)
        
        k_prefix  = output_tuple[0]
        kv_prefix = output_tuple[1]
        
        dec_outputs = output_tuple[2]
        dec_logits  = self.p_decoder(dec_outputs)
        return k_prefix, kv_prefix, dec_logits
    
    def infer(self, x):
        input_len = tf.shape(x)[1]
        infer_ids = [tf.expand_dims(x[:, 0], axis=1)]
        
        for step in range(self.seq_len):
            tmp_inputs = tf.concat(infer_ids, axis=1)
            tmp_tuple  = self.call(tmp_inputs, training=False)
            
            tmp_logit = tmp_tuple[2][:, -1, :]
            tmp_index = tf.cond(
                step < (input_len-1), 
                lambda: x[:, step+1], 
                lambda: tf.argmax(
                    tmp_logit, axis=-1, output_type=tf.int32))
            infer_ids.append(tf.expand_dims(tmp_index, axis=1))
        return tf.concat(infer_ids, axis=1)
    
    # More efficient inference. #
    def infer_v1(self, x):
        input_len = tf.shape(x)[1]
        infer_ids = [tf.expand_dims(x[:, 0], axis=1)]
        
        k_prev  = None
        kv_prev = None
        for step in range(self.seq_len):
            x_prev = infer_ids[-1]
            tmp_tuple  = self.call(
                x_prev, l_st=step, k_prev=k_prev, 
                kv_prev=kv_prev, training=False)
            tmp_logits = tmp_tuple[2]
            
            k_prev  = tmp_tuple[0]
            kv_prev = tmp_tuple[1]
            
            tmp_logit = tmp_logits[:, -1, :]
            tmp_index = tf.cond(
                step < (input_len-1), 
                lambda: x[:, step+1], 
                lambda: tf.argmax(
                    tmp_logit, axis=-1, output_type=tf.int32))
            infer_ids.append(tf.expand_dims(tmp_index, axis=1))
        return tf.concat(infer_ids, axis=1)
        

