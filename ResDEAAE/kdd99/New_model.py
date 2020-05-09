import tensorflow as tf


class BiWGAN(object):
    """resnet + adversarial autoencoder"""
    def __init__(self, input_dim, method, weight, degree, latent_dim=32, dis_fm_dim=128):
        # ==== Parameters ====
        init_kernel = tf.contrib.layers.xavier_initializer()
        ema_decay = 0.9999
        LAMBDA = 10
        LAMBDA_2 = 2
        leaky_relu_slope = 0.1
        dropout_rate = 0.2

        def get_getter(ema):
            """to use neural network with moving average variables"""
            def ema_getter(getter, name, *args, **kwargs):
                var = getter(name, *args, **kwargs)
                ema_var = ema.average(var)
                return ema_var if ema_var else var
            return ema_getter

        # ==== Encoder ====
        def encoder(x_inp, reuse=False, getter=None):
            with tf.variable_scope('encoder', reuse=reuse, custom_getter=getter):
                with tf.variable_scope('layer_1'):
                    enc1 = tf.layers.dense(x_inp, units=4*latent_dim, kernel_initializer=init_kernel, name='fc')
                    enc1 = tf.nn.leaky_relu(enc1, alpha=leaky_relu_slope)

                with tf.variable_scope('layer_2'):
                    enc2 = tf.layers.dense(enc1, units=2*latent_dim, kernel_initializer=init_kernel, name='fc')
                    enc2 = tf.nn.leaky_relu(enc2, alpha=leaky_relu_slope)

                with tf.variable_scope('layer_3'):
                    net = tf.layers.dense(enc2, units=latent_dim, kernel_initializer=init_kernel, name='fc')
            return net, (enc2, enc1)

        # ==== Generator(Decoder) ====
        def generator(z_inp, encs, reuse=False, getter=None):
            with tf.variable_scope('generator', reuse=reuse, custom_getter=getter):
                with tf.variable_scope('layer_1'):
                    dec1 = tf.layers.dense(z_inp, units=2*latent_dim, kernel_initializer=init_kernel, name='fc')
                    dec1 = tf.add(dec1, encs[0])
                    dec1 = tf.nn.leaky_relu(dec1, alpha=leaky_relu_slope)

                with tf.variable_scope('layer_2'):
                    dec2 = tf.layers.dense(dec1, units=4*latent_dim, kernel_initializer=init_kernel, name='fc')
                    dec2 = tf.add(dec2, encs[1])
                    dec2 = tf.nn.leaky_relu(dec2, alpha=leaky_relu_slope)

                with tf.variable_scope('layer_3'):
                    net = tf.layers.dense(dec2, units=input_dim, kernel_initializer=init_kernel, name='fc')
            return net

        # ==== Discriminator ====
        def discriminator(z_inp, x_inp, is_training=False, reuse=False, getter=None):
            with tf.variable_scope('discriminator', reuse=reuse, custom_getter=getter):
                # D(x)
                with tf.variable_scope('x_layer_1'):
                    x = tf.layers.dense(x_inp, units=dis_fm_dim, kernel_initializer=init_kernel, name='fc')
                    x = tf.nn.leaky_relu(x, alpha=leaky_relu_slope)

                # D(z)
                with tf.variable_scope('z_fc_1'):
                    z = tf.layers.dense(z_inp, units=dis_fm_dim, kernel_initializer=init_kernel)
                    z = tf.nn.leaky_relu(z, alpha=leaky_relu_slope)

                # D(x,z)
                y = tf.concat([x, z], axis=1)

                with tf.variable_scope('y_fc_1'):
                    y = tf.layers.dense(y, units=dis_fm_dim, kernel_initializer=init_kernel)
                    y = tf.nn.leaky_relu(y, alpha=leaky_relu_slope)
                    # y = tf.layers.dropout(y, rate=dropout_rate, training=is_training, name='dropout')

                intermediate_layer = y

                with tf.variable_scope('y_fc_logits'):
                    logits = tf.layers.dense(y, units=1, kernel_initializer=init_kernel, name='fc')

            return logits, intermediate_layer

        # ==== Placeholders====
        self.input_pl = tf.placeholder(tf.float32, shape=[None, input_dim], name='input')
        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name='is_training_pl')
        self.lr = tf.placeholder(tf.float32, shape=[], name='lr_pl')

        # ================== Building training graph ==================
        batch_size = tf.shape(self.input_pl)[0]
        with tf.variable_scope('encoder_model'):
            z_gen, encs_gen = encoder(self.input_pl)

        with tf.variable_scope('generator_model'):
            z = tf.random_normal([batch_size, latent_dim])
            x_gen = generator(z, encs_gen)

        # ======================
        with tf.variable_scope('encoder_model'):
            z_rct, _ = encoder(x_gen, reuse=True)
        with tf.variable_scope('generator_model'):
            x_rct = generator(z_gen, encs_gen, reuse=True)
        # ======================

        with tf.variable_scope('discriminator_model'):
            logits_dis_enc, fm_layer_inp = discriminator(z_gen,
                                                         self.input_pl,
                                                         is_training=self.is_training_pl)
            logits_dis_gen, fm_layer_rct = discriminator(z,
                                                         x_gen,
                                                         is_training=self.is_training_pl,
                                                         reuse=True)

            # interpolation
            alpha = tf.random_uniform(shape=[batch_size, 1], minval=0., maxval=1.)   # increase range
            interpolates_x = alpha * self.input_pl + ((1 - alpha) * x_gen)
            interpolates_z = alpha * z_gen + ((1 - alpha) * z)
            disc_interpolates = discriminator(z_inp=interpolates_z,
                                              x_inp=interpolates_x,
                                              is_training=False,
                                              reuse=True)
            # ct
            logits_dis_enc_, fm_layer_inp_ = discriminator(z_inp=z_gen,
                                                           x_inp=self.input_pl,
                                                           is_training=True,
                                                           reuse=True)

        with tf.name_scope('loss_functions'):
            # Discriminator
            gradients_x = tf.gradients(disc_interpolates[0], [interpolates_x])[0]
            gradients_z = tf.gradients(disc_interpolates[0], [interpolates_z])[0]
            slope = tf.sqrt(tf.reduce_sum(tf.square(tf.concat([gradients_x, gradients_z], axis=-1)), axis=1))
            gradient_penalty = tf.reduce_mean((slope - 1.)**2)

            ct = tf.square(logits_dis_enc - logits_dis_enc_)
            ct += 0.1 * tf.reduce_mean(tf.square(fm_layer_inp - fm_layer_inp_), axis=1)
            ct_ = tf.reduce_mean(tf.maximum(0., ct))

            self.loss_discriminator = tf.reduce_mean(logits_dis_gen) - tf.reduce_mean(logits_dis_enc) + \
                                      LAMBDA*gradient_penalty + LAMBDA_2*ct_

            self.loss_generator = -tf.reduce_mean(logits_dis_gen)  # Generator

            self.loss_encoder = tf.reduce_mean(logits_dis_enc)  # Encoder

            loss_reconstruction = tf.losses.mean_squared_error(self.input_pl, x_rct) + \
                                  tf.losses.mean_squared_error(z, z_rct)   # Reconstruction error

        # ==== Optimizers ====
        with tf.name_scope('optimizers'):
            # Control op dependencies for batch normalization and trainable variables
            tvars = tf.trainable_variables()
            dvars = [var for var in tvars if 'discriminator_model' in var.name]
            g_evars = [var for var in tvars if 'generator_model' in var.name or 'encoder_model' in var.name]

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            update_ops_dis = [x for x in update_ops if 'discriminator_model' in x.name]
            update_ops_gen_enc = [x for x in update_ops if 'generator_model' in x.name or 'encoder_model' in x.name]

            optimizer_dis = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9, name='dis_optimizer')
            optimizer_gen_enc = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.9,
                                                       name='gen_enc_optimizer')

            with tf.control_dependencies(update_ops_dis):
                dis_op = optimizer_dis.minimize(self.loss_discriminator, var_list=dvars)
            with tf.control_dependencies(update_ops_gen_enc):
                gen_enc_op = optimizer_gen_enc.minimize(self.loss_generator+self.loss_encoder+0.01*loss_reconstruction,
                                                        var_list=g_evars)

            # Exponential Moving Average for better estimation performance
            dis_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
            maintain_averages_op_dis = dis_ema.apply(dvars)
            with tf.control_dependencies([dis_op]):
                self.train_dis_op = tf.group(maintain_averages_op_dis)

            gen_enc_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
            maintain_averages_op_gen_enc = gen_enc_ema.apply(g_evars)
            with tf.control_dependencies([gen_enc_op]):
                self.train_gen_enc_op = tf.group(maintain_averages_op_gen_enc)

        # ================== Build testing graph ==================
        with tf.variable_scope('encoder_model'):
            z_gen_ema, encs_gen_ema = encoder(self.input_pl,
                                              reuse=True, getter=get_getter(gen_enc_ema))

        with tf.variable_scope('generator_model'):
            x_gen_ema = generator(z_gen_ema, encs_gen_ema,
                                  reuse=True, getter=get_getter(gen_enc_ema))

        with tf.variable_scope('discriminator_model'):
            logits_dis_enc_ema, fm_layer_inp_ema = discriminator(z_gen_ema,
                                                                 self.input_pl,
                                                                 is_training=self.is_training_pl,
                                                                 getter=get_getter(dis_ema),
                                                                 reuse=True)
            logits_dis_gen_ema, fm_layer_rct_ema = discriminator(z_gen_ema,
                                                                 x_gen_ema,
                                                                 is_training=self.is_training_pl,
                                                                 getter=get_getter(dis_ema),
                                                                 reuse=True)
        with tf.name_scope('Testing'):
            with tf.variable_scope('Reconstruction_loss'):
                delta = self.input_pl - x_gen_ema
                delta_flat = tf.layers.flatten(delta)
                self.gen_score = tf.squeeze(tf.norm(delta_flat, ord=degree, axis=1, keepdims=False, name='rct_loss'))

            with tf.variable_scope('Discriminator_loss'):
                if method == 'cross-e':
                    self.dis_score = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_dis_gen_ema),
                                                                             logits=logits_dis_gen_ema)
                elif method == 'fm':
                    fm = fm_layer_inp_ema - fm_layer_rct_ema
                    fm = tf.layers.flatten(fm)
                    self.dis_score = tf.norm(fm, ord=degree, axis=1, keepdims=False, name='d_loss')
                self.dis_score = tf.squeeze(self.dis_score)

            with tf.variable_scope('Score'):
                self.list_scores = weight * self.gen_score + (1 - weight) * self.dis_score

    def train(self, sess, batch_data, learning_rate, train_d=True):
        # Train discriminator
        if train_d:
            loss_dis, loss_gen, loss_enc, _ = sess.run(
                [self.loss_discriminator, self.loss_generator, self.loss_encoder, self.train_dis_op],
                feed_dict={self.input_pl: batch_data,
                           self.is_training_pl: True,
                           self.lr: learning_rate})

        # Train generator and encoder
        else:
            loss_dis, loss_gen, loss_enc, _ = sess.run(
                [self.loss_discriminator, self.loss_generator, self.loss_encoder,
                 self.train_gen_enc_op],
                feed_dict={self.input_pl: batch_data,
                           self.is_training_pl: True,
                           self.lr: learning_rate})

        return loss_dis, loss_gen, loss_enc

    def eval(self, sess, batch_test_data):
        ano_score, rct_score, dis_score = sess.run([self.list_scores, self.gen_score, self.dis_score],
                                                   feed_dict={self.input_pl: batch_test_data,
                                                              self.is_training_pl: False})
        return ano_score, rct_score, dis_score

    @staticmethod
    def save(sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    @staticmethod
    def restore(sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)

