import tensorflow as tf


class BiGAN(object):
    def __init__(self, count_list, method, weight=0.9, degree=1, latent_dim=32, dis_fm_dim=128):
        # ==== Parameters ====
        init_kernel = tf.contrib.layers.xavier_initializer()
        ema_decay = 0.9999
        emb_dim = 32
        bucket_size = 1e5
        leak_relu_slope = 0.1
        dropout_rate = 0.5
        [ip_count, app_count, dev_count, os_count, chan_count,
         sec_count, minu_count, hour_count, day_count, dfw_count] = count_list

        def get_getter(ema):
            """to use neural network with moving average variables"""
            def ema_getter(getter, name, *args, **kwargs):
                var = getter(name, *args, **kwargs)
                ema_var = ema.average(var)
                return ema_var if ema_var else var
            return ema_getter

        def embedding(inputs, is_training=False, reuse=False, getter=None):
            with tf.variable_scope('embedding_layer', reuse=reuse, custom_getter=getter):
                [ip, app, dev, os, chan, sec, minu, hour, day, dfw] = inputs
                # ip = tf.string_to_hash_bucket_fast(input=ip, num_buckets=bucket_size)

                ip_emb_w = tf.get_variable('ip_emb_w', [ip_count, 2*emb_dim], initializer=init_kernel)
                app_emb_w = tf.get_variable('app_emb_w', [app_count, emb_dim], initializer=init_kernel)
                dev_emb_w = tf.get_variable('dev_emb_w', [dev_count, emb_dim], initializer=init_kernel)
                os_emb_w = tf.get_variable('os_emb_w', [os_count, emb_dim], initializer=init_kernel)
                chan_emb_w = tf.get_variable('chan_emb_w', [chan_count, emb_dim], initializer=init_kernel)
                sec_emb_w = tf.get_variable('sec_emb_w', [sec_count, emb_dim], initializer=init_kernel)
                minu_emb_w = tf.get_variable('minu_emb_w', [minu_count, emb_dim], initializer=init_kernel)
                hour_emb_w = tf.get_variable('hour_emb_w', [hour_count, emb_dim//2], initializer=init_kernel)
                day_emb_w = tf.get_variable('day_emb_w', [day_count, emb_dim//2], initializer=init_kernel)
                dfw_emb_w = tf.get_variable('dfw_emb_w', [dfw_count, emb_dim//2], initializer=init_kernel)

                ip_emb = tf.nn.embedding_lookup(ip_emb_w, ip)
                app_emb = tf.nn.embedding_lookup(app_emb_w, app)
                dev_emb = tf.nn.embedding_lookup(dev_emb_w, dev)
                os_emb = tf.nn.embedding_lookup(os_emb_w, os)
                chan_emb = tf.nn.embedding_lookup(chan_emb_w, chan)
                sec_emb = tf.nn.embedding_lookup(sec_emb_w, sec)
                minu_emb = tf.nn.embedding_lookup(minu_emb_w, minu)
                hour_emb = tf.nn.embedding_lookup(hour_emb_w, hour)
                day_emb = tf.nn.embedding_lookup(day_emb_w, day)
                dfw_emb = tf.nn.embedding_lookup(dfw_emb_w, dfw)

                emb_inp = tf.concat([ip_emb, app_emb, dev_emb, os_emb, chan_emb,
                                     sec_emb, minu_emb, hour_emb, day_emb, dfw_emb], axis=-1)
            return emb_inp

        def encoder(emb_inp, is_training=False, reuse=False, getter=None):
            with tf.variable_scope('encoder', reuse=reuse, custom_getter=getter):
                with tf.variable_scope('layer_1'):
                    net = tf.layers.dense(emb_inp, units=8*latent_dim, kernel_initializer=init_kernel, name='fc')
                    net = tf.nn.leaky_relu(net, alpha=leak_relu_slope)

                with tf.variable_scope('layer_2'):
                    net = tf.layers.dense(net, units=4*latent_dim, kernel_initializer=init_kernel, name='fc')
                    net = tf.nn.leaky_relu(net, alpha=leak_relu_slope)

                with tf.variable_scope('layer_3'):
                    net = tf.layers.dense(net, units=2*latent_dim, kernel_initializer=init_kernel, name='fc')
                    net = tf.nn.leaky_relu(net, alpha=leak_relu_slope)

                with tf.variable_scope('logits'):
                    net = tf.layers.dense(net, units=latent_dim, kernel_initializer=init_kernel, name='fc')
            return net

        def generator(z_inp, x_inp_dim, is_training=False, reuse=False, getter=None):
            with tf.variable_scope('generator', reuse=reuse, custom_getter=getter):
                with tf.variable_scope('layer_1'):
                    net = tf.layers.dense(z_inp, units=2*latent_dim, kernel_initializer=init_kernel, name='fc')
                    net = tf.nn.leaky_relu(net, alpha=leak_relu_slope)

                with tf.variable_scope('layer_2'):
                    net = tf.layers.dense(net, units=4*latent_dim, kernel_initializer=init_kernel, name='fc')
                    net = tf.nn.leaky_relu(net, alpha=leak_relu_slope)

                with tf.variable_scope('layer_3'):
                    net = tf.layers.dense(net, units=8*latent_dim, kernel_initializer=init_kernel, name='fc')
                    net = tf.nn.leaky_relu(net, alpha=leak_relu_slope)

                with tf.variable_scope('logits'):
                    net = tf.layers.dense(net, units=x_inp_dim, kernel_initializer=init_kernel, name='fc')
            return net

        def discriminator(z_inp, x_inp, is_training=False, reuse=False, getter=None):
            with tf.variable_scope('discriminator', reuse=reuse, custom_getter=getter):
                # D(x)
                with tf.variable_scope('x_fc_1'):
                    x = tf.layers.dense(x_inp, units=4*dis_fm_dim, kernel_initializer=init_kernel, name='fc')
                    x = tf.nn.leaky_relu(x, alpha=leak_relu_slope)
                    x = tf.layers.dropout(x, rate=dropout_rate, training=is_training, name='dropout')
                # D(z)
                with tf.variable_scope('z_fc_1'):
                    z = tf.layers.dense(z_inp, units=4*dis_fm_dim, kernel_initializer=init_kernel)
                    z = tf.nn.leaky_relu(z, alpha=leak_relu_slope)
                    z = tf.layers.dropout(z, rate=dropout_rate, training=is_training, name='dropout')
                # concatenate
                y = tf.concat([x, z], axis=-1)
                with tf.variable_scope('y_fc_1'):
                    y = tf.layers.dense(y, units=4*dis_fm_dim, kernel_initializer=init_kernel)
                    y = tf.nn.leaky_relu(y, alpha=leak_relu_slope)
                    y = tf.layers.dropout(y, rate=dropout_rate, training=is_training, name='dropout')

                with tf.variable_scope('y_fc_2'):
                    y = tf.layers.dense(y, units=2*dis_fm_dim, kernel_initializer=init_kernel)
                    y = tf.nn.leaky_relu(y, alpha=leak_relu_slope)
                    y = tf.layers.dropout(y, rate=dropout_rate, training=is_training, name='dropout')

                with tf.variable_scope('y_fc_fm'):
                    y = tf.layers.dense(y, units=dis_fm_dim, kernel_initializer=init_kernel)
                    y = tf.nn.leaky_relu(y, alpha=leak_relu_slope)
                    y = tf.layers.dropout(y, rate=dropout_rate, training=is_training, name='dropout')

                intermediate_layer = y
                with tf.variable_scope('y_fc_logits'):
                    logits = tf.layers.dense(y, units=1, kernel_initializer=init_kernel, name='fc')
            return logits, intermediate_layer

        # ==== Placeholders ====
        # self.ip_pl = tf.placeholder(tf.string, shape=[None, ], name='ip')

        self.ip_pl = tf.placeholder(tf.int32, shape=[None, ], name='ip')
        self.app_pl = tf.placeholder(tf.int32, shape=[None, ], name='app')
        self.device_pl = tf.placeholder(tf.int32, shape=[None, ], name='device')
        self.os_pl = tf.placeholder(tf.int32, shape=[None, ], name='os')
        self.channel_pl = tf.placeholder(tf.int32, shape=[None, ], name='channel')
        self.sec_pl = tf.placeholder(tf.int32, shape=[None, ], name='second')
        self.minu_pl = tf.placeholder(tf.int32, shape=[None, ], name='minute')
        self.hour_pl = tf.placeholder(tf.int32, shape=[None, ], name='hour')
        self.day_pl = tf.placeholder(tf.int32, shape=[None, ], name='day')
        self.dayofweek_pl = tf.placeholder(tf.int32, shape=[None, ], name='dayofweek')

        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name='is_training_pl')
        self.lr = tf.placeholder(tf.float32, shape=[], name='lr_pl')

        # ================== Building Training Graph ==================
        with tf.variable_scope('encoder_model'):
            features = [self.ip_pl, self.app_pl, self.device_pl, self.os_pl, self.channel_pl,
                        self.sec_pl, self.minu_pl, self.hour_pl, self.day_pl, self.dayofweek_pl]
            emb_inp = embedding(inputs=features, is_training=self.is_training_pl)
            z_gen = encoder(emb_inp=emb_inp, is_training=self.is_training_pl)
            emb_inp_dim = emb_inp.get_shape().as_list()[-1]

        with tf.variable_scope('generator_model'):
            z = tf.random.normal([tf.shape(self.ip_pl)[0], latent_dim])
            x_gen = generator(z_inp=z, x_inp_dim=emb_inp_dim, is_training=self.is_training_pl)

        with tf.variable_scope('discriminator_model'):
            logits_dis_enc, fm_layer_inp = discriminator(z_inp=z_gen,
                                                         x_inp=emb_inp,
                                                         is_training=self.is_training_pl)
            logits_dis_gen, fm_layer_rct = discriminator(z_inp=z, x_inp=x_gen,
                                                         is_training=self.is_training_pl,
                                                         reuse=True)

        with tf.name_scope('loss_functions'):
            # Discriminator loss
            loss_dis_enc = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_dis_enc),
                                                        logits=logits_dis_enc))
            loss_dis_gen = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_dis_gen),
                                                        logits=logits_dis_gen))
            self.loss_discriminator = loss_dis_enc + loss_dis_gen

            # Generator loss
            self.loss_generator = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_dis_gen),
                                                        logits=logits_dis_gen))
            # Encoder loss
            self.loss_encoder = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_dis_enc),
                                                        logits=logits_dis_enc))

        # ================== Optimizers ==================
        with tf.name_scope('optimizers'):
            # Control op dependencies for batch normalization and trainable variables
            tvars = tf.trainable_variables()
            self.dvars = [var for var in tvars if 'discriminator_model' in var.name]
            self.gvars = [var for var in tvars if 'generator_model' in var.name]
            self.evars = [var for var in tvars if 'encoder_model' in var.name]
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            update_ops_dis = [x for x in update_ops if 'discriminator_model' in x.name]
            update_ops_gen = [x for x in update_ops if 'generator_model' in x.name]
            update_ops_enc = [x for x in update_ops if 'encoder_model' in x.name]

            optimizer_dis = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, name='dis_optimizer')
            optimizer_gen = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, name='gen_optimizer')
            optimizer_enc = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, name='enc_optimizer')

            with tf.control_dependencies(update_ops_dis):
                dis_op = optimizer_dis.minimize(self.loss_discriminator, var_list=self.dvars)
            with tf.control_dependencies(update_ops_gen):
                gen_op = optimizer_gen.minimize(self.loss_generator, var_list=self.gvars)
            with tf.control_dependencies(update_ops_enc):
                enc_op = optimizer_enc.minimize(self.loss_encoder, var_list=self.evars)

            # Exponential Moving Average for better estimation performance
            dis_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
            maintain_averages_op_dis = dis_ema.apply(self.dvars)
            with tf.control_dependencies([dis_op]):
                self.train_dis_op = tf.group(maintain_averages_op_dis)

            gen_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
            maintain_averages_op_gen = gen_ema.apply(self.gvars)
            with tf.control_dependencies([gen_op]):
                self.train_gen_op = tf.group(maintain_averages_op_gen)

            enc_ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
            maintain_averages_op_enc = enc_ema.apply(self.evars)
            with tf.control_dependencies([enc_op]):
                self.train_enc_op = tf.group(maintain_averages_op_enc)

        # ================== Build Testing Graph ==================
        with tf.variable_scope('encoder_model'):
            features_ema = [self.ip_pl, self.app_pl, self.device_pl, self.os_pl, self.channel_pl,
                            self.sec_pl, self.minu_pl, self.hour_pl, self.day_pl, self.dayofweek_pl]
            emb_inp_ema = embedding(inputs=features_ema,
                                    reuse=True,
                                    getter=get_getter(enc_ema))
            self.z_gen_ema = encoder(emb_inp=emb_inp_ema,
                                     is_training=self.is_training_pl,
                                     getter=get_getter(enc_ema),
                                     reuse=True)
            emb_inp_ema_dim = emb_inp_ema.get_shape().as_list()[-1]

        with tf.variable_scope('generator_model'):
            x_gen_ema = generator(z_inp=self.z_gen_ema,
                                  x_inp_dim=emb_inp_ema_dim,
                                  is_training=self.is_training_pl,
                                  getter=get_getter(gen_ema),
                                  reuse=True)

        with tf.variable_scope('discriminator_model'):
            logits_dis_enc_ema, fm_layer_inp_ema = discriminator(z_inp=self.z_gen_ema,
                                                                 x_inp=emb_inp_ema,
                                                                 is_training=self.is_training_pl,
                                                                 getter=get_getter(dis_ema),
                                                                 reuse=True)
            logits_dis_gen_ema, fm_layer_rct_ema = discriminator(z_inp=self.z_gen_ema,
                                                                 x_inp=x_gen_ema,
                                                                 is_training=self.is_training_pl,
                                                                 getter=get_getter(dis_ema),
                                                                 reuse=True)
        with tf.name_scope('Testing'):
            with tf.variable_scope('Reconstruction_loss'):
                delta = emb_inp_ema - x_gen_ema
                delta_flat = tf.layers.flatten(delta)
                self.gen_score = tf.norm(delta_flat, ord=degree, axis=1,
                                         keepdims=False, name='rct_loss')

            with tf.variable_scope('Discriminator_loss'):
                if method == 'cross-e':
                    self.dis_score = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_dis_gen_ema),
                                                                             logits=logits_dis_gen_ema)
                elif method == 'fm':
                    fm = fm_layer_inp_ema - fm_layer_rct_ema
                    fm = tf.layers.flatten(fm)
                    self.dis_score = tf.norm(fm, ord=degree, axis=1,
                                             keepdims=False, name='d_loss')
                self.dis_score = tf.squeeze(self.dis_score)

            with tf.variable_scope('Score'):
                self.list_scores = weight * self.gen_score + (1 - weight) * self.dis_score

    def train(self, sess, batch_data, learning_rate, train_d=True):
        feed_dict = {self.ip_pl: batch_data[:, 0],  # astype(np.string_)
                     self.app_pl: batch_data[:, 1],
                     self.device_pl: batch_data[:, 2],
                     self.os_pl: batch_data[:, 3],
                     self.channel_pl: batch_data[:, 4],
                     self.sec_pl: batch_data[:, 5],
                     self.minu_pl: batch_data[:, 6],
                     self.hour_pl: batch_data[:, 7],
                     self.day_pl: batch_data[:, 8],
                     self.dayofweek_pl: batch_data[:, 9],
                     self.is_training_pl: True,
                     self.lr: learning_rate}
        # Train discriminator
        if train_d:
            loss_dis, loss_gen, loss_enc, _ = sess.run(
                [self.loss_discriminator, self.loss_generator, self.loss_encoder, self.train_dis_op],
                feed_dict=feed_dict)

        # Train generator and encoder
        else:
            loss_dis, loss_gen, loss_enc, _, _ = sess.run(
                [self.loss_discriminator, self.loss_generator, self.loss_encoder,
                 self.train_gen_op, self.train_enc_op],
                feed_dict=feed_dict)
        return loss_dis, loss_gen, loss_enc

    def eval(self, sess, batch_data):
        ano_score = sess.run(self.list_scores,
                             feed_dict={self.ip_pl: batch_data[:, 0],
                                        self.app_pl: batch_data[:, 1],
                                        self.device_pl: batch_data[:, 2],
                                        self.os_pl: batch_data[:, 3],
                                        self.channel_pl: batch_data[:, 4],
                                        self.sec_pl: batch_data[:, 5],
                                        self.minu_pl: batch_data[:, 6],
                                        self.hour_pl: batch_data[:, 7],
                                        self.day_pl: batch_data[:, 8],
                                        self.dayofweek_pl: batch_data[:, 9],
                                        self.is_training_pl: False})
        return ano_score
    
    def get_lat_repres(self, sess, batch_data):
        lat_repres = sess.run(self.z_gen_ema,
                              feed_dict={self.ip_pl: batch_data[:, 0],
                                         self.app_pl: batch_data[:, 1],
                                         self.device_pl: batch_data[:, 2],
                                         self.os_pl: batch_data[:, 3],
                                         self.channel_pl: batch_data[:, 4],
                                         self.sec_pl: batch_data[:, 5],
                                         self.minu_pl: batch_data[:, 6],
                                         self.hour_pl: batch_data[:, 7],
                                         self.day_pl: batch_data[:, 8],
                                         self.dayofweek_pl: batch_data[:, 9],
                                         self.is_training_pl: False})
        return lat_repres
    
    def get_weights(self):
        return [self.dvars, self.gvars, self.evars]

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
