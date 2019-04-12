import tensorflow as tf 
import numpy as np 
import os
import sys
import matplotlib.pyplot as plt 
import config

from cifarIO import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('tmp/data/',one_hot=True);

seed = 10;
tf.set_random_seed(seed);
random.seed(seed);
'''
For better readability
I will be using same naming conventions as provided in
pseudo code of official paper.
'''

#Used to initialize kernel weights
stddev = 0.02;#99999;

opts = config.config_cifar10

encoder_learning_rate = opts['enc_lr'];
generator_learning_rate = opts['gen_lr'];
discriminator_learning_rate = opts['disc_lr'];
code_discriminator_learning_rate = opts['code_disc_lr'];

batch_size = opts['batch_size'];

n_epoch = opts['n_epoch'];
z_dim = opts['z_dim'];

tfd = tf.contrib.distributions

#model_params
n_inputs = 28*28; #as images are of 28 x 28 dimension
img_height = opts['img_height'];
img_width = opts['img_width'];
num_channels = opts['num_channels'];
n_outputs = 10;

#Placeholder
X = tf.placeholder(tf.float32,[None,img_height,img_width,num_channels]);
Y = tf.placeholder(tf.float32,[None,n_outputs]);

def prior_z(latent_dim):
        z_mean = tf.zeros(latent_dim);
        z_var = tf.ones(latent_dim);
        return tfd.MultivariateNormalDiag(z_mean,z_var);

def encoder(X,isTrainable=True,reuse=False,name='eta_encoder'):
    with tf.variable_scope(name) as scope:
        #encoder_activations = {};
        if reuse:
            scope.reuse_variables();

        #32x32x3
        conv1 = tf.layers.conv2d(X,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=64,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv1_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv1 = tf.layers.batch_normalization(conv1,training=isTrainable,reuse=reuse,name='bn_1');
        conv1 = tf.nn.relu(conv1,name='relu_conv_1');
        
        #16x16x64
        conv2 = tf.layers.conv2d(conv1,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=128,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv2_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv2 = tf.layers.batch_normalization(conv2,training=isTrainable,reuse=reuse,name='bn_2');
        conv2 = tf.nn.relu(conv2,name='relu_conv_2');
        
        #8x8x128
        conv3 = tf.layers.conv2d(conv2,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=256,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv3_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv3 = tf.layers.batch_normalization(conv3,training=isTrainable,reuse=reuse,name='bn_3');
        conv3 = tf.nn.relu(conv3,name='relu_conv_3');
    
        #4x4x256
        conv4 = tf.layers.conv2d(conv3,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=512,kernel_size=[5,5],padding='SAME',strides=(1,1),name='enc_conv4_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv4 = tf.layers.batch_normalization(conv4,training=isTrainable,reuse=reuse,name='bn_4');
        conv4 = tf.nn.relu(conv4,name='relu_conv_4');
        
        #4x4x512
        conv4_flattened = tf.layers.flatten(conv4);
        
        latent_code = tf.layers.dense(conv4_flattened,z_dim,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),activation=None,name='enc_latent_space',trainable=isTrainable,reuse=reuse);
        return latent_code;

def generator(z_sample,isTrainable=True,reuse=False,name='theta_generator'):
    with tf.variable_scope(name) as scope:

        if reuse:
            scope.reuse_variables();

        z_sample = tf.layers.dense(z_sample,4*4*512,activation=None,trainable=isTrainable,reuse=reuse,name='dec_dense_fc_first_layer',kernel_initializer=tf.truncated_normal_initializer(stddev=stddev));
        z_sample = tf.layers.batch_normalization(z_sample,training=isTrainable,reuse=reuse,name='bn_0');
        z_sample = tf.nn.relu(z_sample);
        z_sample = tf.reshape(z_sample,[-1,4,4,512]);
        #4x4x512

        deconv1 = tf.layers.conv2d_transpose(z_sample,kernel_initializer=tf.random_normal_initializer(stddev=stddev),filters=256,kernel_size=[5,5],padding='SAME',activation=None,strides=(2,2),name='dec_deconv1_layer',trainable=isTrainable,reuse=reuse); # 16x16
        deconv1 = tf.layers.batch_normalization(deconv1,training=isTrainable,reuse=reuse,name='bn_1');
        deconv1 = tf.nn.relu(deconv1,name='relu_deconv_1');
         
        #8x8x256
        deconv2 = tf.layers.conv2d_transpose(deconv1,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=128,kernel_size=[5,5],padding='SAME',activation=None,strides=(2,2),name='dec_deconv2_layer',trainable=isTrainable,reuse=reuse); # 16x16
        deconv2 = tf.layers.batch_normalization(deconv2,training=isTrainable,reuse=reuse,name='bn_2');
        deconv2 = tf.nn.relu(deconv2,name='relu_deconv_2');
        
        #16x16x128
        deconv3 = tf.layers.conv2d_transpose(deconv2,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=64,kernel_size=[5,5],padding='SAME',activation=None,strides=(2,2),name='dec_deconv3_layer',trainable=isTrainable,reuse=reuse); # 16x16
        deconv3 = tf.layers.batch_normalization(deconv3,training=isTrainable,reuse=reuse,name='bn_3');
        deconv3 = tf.nn.relu(deconv3,name='relu_deconv_3');
        
        #32x32x64 
        deconv4 = tf.layers.conv2d_transpose(deconv3,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=3,kernel_size=[5,5],padding='SAME',activation=None,strides=(1,1),name='dec_deconv4_layer',trainable=isTrainable,reuse=reuse); # 16x16    
        #deconv4 = tf.layers.dropout(deconv4,rate=keep_prob,training=True);
        deconv4 = tf.nn.tanh(deconv4);
        #32x32x3
        
        deconv_4_reshaped = tf.reshape(deconv4,[-1,img_height,img_width,num_channels]);
        return deconv_4_reshaped;

def discriminator(X,isTrainable=True,reuse=False,name='phi_discriminator'):
    with tf.variable_scope(name) as scope:

        if reuse:
            scope.reuse_variables();

        #32x32x3 --> means size of input before applying conv1
        conv1 = tf.layers.conv2d(X,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=64,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv1_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        #conv1 = tf.layers.batch_normalization(conv1,training=isTrainable,reuse=reuse,name='bn_1');
        conv1 = tf.nn.leaky_relu(conv1,name='leaky_relu_conv_1');
        
        #16x16x64
        conv2 = tf.layers.conv2d(conv1,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=128,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv2_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv2 = tf.layers.batch_normalization(conv2,training=isTrainable,reuse=reuse,name='bn_2');
        conv2 = tf.nn.leaky_relu(conv2,name='leaky_relu_conv_2');
        
        #8x8x128
        conv3 = tf.layers.conv2d(conv2,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=256,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv3_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv3 = tf.layers.batch_normalization(conv3,training=isTrainable,reuse=reuse,name='bn_3');
        conv3 = tf.nn.leaky_relu(conv3,name='leaky_relu_conv_3');
    
        #4x4x256
        conv4 = tf.layers.conv2d(conv3,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=512,kernel_size=[5,5],padding='SAME',strides=(1,1),name='enc_conv4_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv4 = tf.layers.batch_normalization(conv4,training=isTrainable,reuse=reuse,name='bn_4');
        conv4 = tf.nn.leaky_relu(conv4,name='leaky_relu_conv_4');
        
        #4x4x512
        conv4_flattened = tf.layers.flatten(conv4);
        
        output_disc = tf.layers.dense(conv4_flattened,1,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),activation=None,name='dis_fc_layer',trainable=isTrainable,reuse=reuse);
        
        return output_disc;

def code_discriminator(Z,isTrainable=True,reuse=False,name='omega_code_discriminator'):
    with tf.variable_scope(name) as scope:

        if reuse:
            scope.reuse_variables();

        fc1 = tf.layers.dense(Z,750,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),activation=None,name='code_dis_fc_layer_1',trainable=isTrainable,reuse=reuse);
        #fc1 = tf.layers.batch_normalization(fc1,training=isTrainable,reuse=reuse,name='bn_1');
        fc1 = tf.nn.leaky_relu(fc1);

        fc2 = tf.layers.dense(fc1,750,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),activation=None,name='code_dis_fc_layer_2',trainable=isTrainable,reuse=reuse);
        #fc2 = tf.layers.batch_normalization(fc2,training=isTrainable,reuse=reuse,name='bn_2');
        fc2 = tf.nn.leaky_relu(fc2);

        # fc3 = tf.layers.dense(fc2,2,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),activation=None,name='code_dis_fc_layer_3',trainable=isTrainable,reuse=reuse);
        # #fc3 = tf.layers.batch_normalization(fc3,training=isTrainable,reuse=reuse,name='bn_3');
        # fc3 = tf.nn.leaky_relu(fc3);

        logits = tf.layers.dense(fc2,1,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),activation=None,name='code_dis_fc_layer_4',trainable=isTrainable,reuse=reuse);

        return logits;

def RD_phi(D_x):
    # D_x: logits of discriminator D
    non_sat_term = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_x, labels=tf.ones_like(D_x)));
    sat_term = -1 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_x, labels = tf.zeros_like(D_x)));
    return non_sat_term + sat_term;

def RC_w(C_z):
    # C_x: logits of code discriminator C
    non_sat_term = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=C_z, labels=tf.ones_like(C_z)));
    sat_term = -1 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=C_z, labels = tf.zeros_like(C_z)));
    return non_sat_term + sat_term;



prior_dist = prior_z(z_dim);

z_prime = prior_dist.sample(batch_size);
# z_hat ~ Q(Z|X) variational distribution parametrized by encoder network
z_hat = encoder(X);
z_hat_test = encoder(X,isTrainable=False,reuse=True);
x_hat = generator(z_hat);
x_hat_test = generator(z_hat_test,isTrainable=False,reuse=True);
x_hat_test_padded = tf.image.resize_image_with_crop_or_pad(x_hat_test,33,33);
x_prime = generator(z_prime,reuse=True);

z_hat_logits = code_discriminator(z_hat);
z_prime_logits = code_discriminator(z_prime,reuse=True);

x_hat_logits = discriminator(x_hat);
x_prime_logits = discriminator(x_prime,reuse=True);
x_logits = discriminator(X,reuse=True);

lamda = opts['lamda'];
l1_recons_loss = lamda*tf.reduce_mean(tf.abs(X - x_hat));

encoder_loss = l1_recons_loss + RC_w(z_hat_logits);

generator_loss = l1_recons_loss + RD_phi(x_hat_logits) + RD_phi(x_prime_logits);

disc_real_term = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logits, labels=tf.ones_like(x_logits)));
disc_fake_recons_term = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat_logits, labels=tf.zeros_like(x_hat_logits)));
disc_fake_sample_term = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_prime_logits, labels=tf.zeros_like(x_prime_logits)));

discriminator_loss = disc_real_term + disc_fake_recons_term + disc_fake_sample_term;

code_disc_real_term = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z_prime_logits, labels=tf.ones_like(z_prime_logits)));
code_disc_fake_term = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z_hat_logits, labels=tf.zeros_like(z_hat_logits)));

code_discriminator_loss = code_disc_real_term + code_disc_fake_term;

eta_encoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='eta_encoder');
theta_generator_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='theta_generator');
phi_discriminator_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='phi_discriminator');
omega_code_discriminator_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='omega_code_discriminator');

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS);
with tf.control_dependencies(update_ops):
    encoder_optimizer = tf.train.AdamOptimizer(learning_rate=encoder_learning_rate,beta1=0.5,beta2=0.9);
    encoder_gradsVars = encoder_optimizer.compute_gradients(encoder_loss, eta_encoder_params);
    encoder_train_optimizer = encoder_optimizer.apply_gradients(encoder_gradsVars);

    generator_optimizer = tf.train.AdamOptimizer(learning_rate=generator_learning_rate,beta1=0.5,beta2=0.9);
    generator_gradsVars = generator_optimizer.compute_gradients(generator_loss, theta_generator_params);
    generator_train_optimizer = generator_optimizer.apply_gradients(generator_gradsVars);

    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=discriminator_learning_rate,beta1=0.5,beta2=0.9);
    discriminator_gradsVars = discriminator_optimizer.compute_gradients(discriminator_loss, phi_discriminator_params);
    discriminator_train_optimizer = discriminator_optimizer.apply_gradients(discriminator_gradsVars);

    code_discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=code_discriminator_learning_rate,beta1=0.5,beta2=0.9);
    code_discriminator_gradsVars = code_discriminator_optimizer.compute_gradients(code_discriminator_loss, omega_code_discriminator_params);
    code_discriminator_train_optimizer = code_discriminator_optimizer.apply_gradients(code_discriminator_gradsVars);

tf.summary.scalar("encoder_loss",encoder_loss);
tf.summary.scalar("discriminator_loss",discriminator_loss);
tf.summary.scalar("generator_loss",generator_loss);
tf.summary.scalar("code_discriminator_loss",code_discriminator_loss);
tf.summary.scalar("l1_recons_loss",l1_recons_loss);

all_gradsVars = [encoder_gradsVars, generator_gradsVars, discriminator_gradsVars, code_discriminator_gradsVars];

for grad_vars in all_gradsVars:
    for g,v in grad_vars:  
        tf.summary.histogram(v.name,v)
        tf.summary.histogram(v.name+str('grad'),g)

merged_all = tf.summary.merge_all();
log_directory = 'myAlpha-GAN-dir';
model_directory='myAlpha-GAN-model_dir';
output_directory = 'op/';
train_output_directory = 'train-op/';

all_directories  = [log_directory, model_directory, output_directory, train_output_directory];

for direc in all_directories:
    if not os.path.exists(direc):
        os.makedirs(direc);
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory);
# if not os.path.exists(log_directory):
#     os.makedirs(log_directory);
# if not os.path.exists(model_directory):
#     os.makedirs(model_directory); 

# all_train_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES);
# print('---------------trainable params--------------');
# for p in all_train_params:
#     print (p);
# print('--------------------------------------------');

def train():

    X_train = load_data();
    X_train = normalize_image(X_train);

    X_test = load_test_data();
    X_test = normalize_image(X_test);

    n_batches = X_train.shape[0]/batch_size;#mnist.train.num_examples/batch_size;
    n_batches = int(n_batches);

    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer());
        print('n_batches : ',n_batches,' when batch_size: ',batch_size);
        #for tensorboard
        saver = tf.train.Saver(max_to_keep=3);
        writer = tf.summary.FileWriter(log_directory,sess.graph);
        iterations = 0;
        
        for epoch in range(n_epoch):
            for batch in range(n_batches):
                iterations += 1;
                
                #Train Encoder
                i=2;
                for i in range(i):
                    X_batch = get_cifar_batch(X_train,batch_size);
                    fd = {X: X_batch};
                    _,enc_loss= sess.run([encoder_train_optimizer,encoder_loss],feed_dict = fd);

                #Train Generator
                j=2;
                for i in range(j):
                    X_batch = get_cifar_batch(X_train,batch_size);
                    fd = {X: X_batch};
                    _,gen_loss= sess.run([generator_train_optimizer,generator_loss],feed_dict = fd);

                #Train Discriminator
                k=1;
                for i in range(k):
                    X_batch = get_cifar_batch(X_train,batch_size);
                    fd = {X: X_batch};
                    _,disc_loss= sess.run([discriminator_train_optimizer,discriminator_loss],feed_dict = fd);

                #Train Code Discriminator
                l=1;
                for i in range(l):
                    X_batch = get_cifar_batch(X_train,batch_size);
                    fd = {X: X_batch};
                    _,code_disc_loss,merged= sess.run([code_discriminator_train_optimizer,code_discriminator_loss,merged_all],feed_dict = fd);

                if(iterations%20==0):
                    writer.add_summary(merged,iterations);

                if(batch%200 == 0):
                    print('Batch #',batch,' done!');

                #break;

            if(epoch%2==0):

                num_val_img = 25;
                batch_X = get_cifar_batch(X_test,batch_size);
                
                recons = sess.run(x_hat_test,feed_dict={X:batch_X});
                recons = np.reshape(recons,[-1,32,32,3]);


                train_batch_X = get_cifar_batch(X_train,batch_size);
                train_recons = sess.run(x_hat_test,feed_dict={X:train_batch_X});
                train_recons = np.reshape(train_recons,[-1,32,32,3]);

                n_gen = 100;
                sample = tf.random_normal([n_gen,z_dim]);
                generations = sess.run(x_hat_test,feed_dict={z_hat_test:sample.eval()});
                generations = np.reshape(generations,[-1,32,32,3]);

                temp_index = -1;
                for s in range(generations.shape[0]):
                    temp_index += 1;
                    generations[temp_index] = denormalize_image(generations[temp_index]);

                temp_index = -1;
                for s in range(batch_X.shape[0]):
                    temp_index += 1;
                    batch_X[temp_index] = denormalize_image(batch_X[temp_index]);

                temp_index = -1;
                for s in range(recons.shape[0]):
                    temp_index += 1;
                    recons[temp_index] = denormalize_image(recons[temp_index]);


                #----------------------- For Training reconstructions----------
                temp_index = -1;
                for s in range(train_batch_X.shape[0]):
                    temp_index += 1;
                    train_batch_X[temp_index] = denormalize_image(train_batch_X[temp_index]);

                temp_index = -1;
                for s in range(train_recons.shape[0]):
                    temp_index += 1;
                    train_recons[temp_index] = denormalize_image(train_recons[temp_index]);

                #----------------------------------------------------------------

                n = 5;
                reconstructed = np.empty((32*n,32*n,3));
                original = np.empty((32*n,32*n,3));
                
                for i in range(n):
                    for j in range(n):
                        original[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32,:] = batch_X[i*n+j];#.reshape([32, 32,3]);
                        reconstructed[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32,:] = recons[i*n+j];
                        #generated_images[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32,:] = generations[i*n+j];

                n = 5;
                train_reconstructed = np.empty((32*n,32*n,3));
                train_original = np.empty((32*n,32*n,3));
                
                for i in range(n):
                    for j in range(n):
                        train_original[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32,:] = train_batch_X[i*n+j];#.reshape([32, 32,3]);
                        train_reconstructed[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32,:] = train_recons[i*n+j];

                n1 = 10;
                generated_images = np.empty((32*n1,32*n1,3));
                for i in range(n1):
                    for j in range(n1):
                        generated_images[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32,:] = generations[i*n1+j];

                print("Original Images");
                plt.figure(figsize=(n, n));
                plt.imshow(original, origin="upper",interpolation='nearest', cmap="gray");
                plt.savefig('op/orig-img-'+str(epoch)+'.png');
                plt.close();

                print("Reconstructed Images");
                plt.figure(figsize=(n, n));
                plt.imshow(reconstructed, origin="upper",interpolation='nearest', cmap="gray");
                plt.savefig('op/recons-img-'+str(epoch)+'.png');
                plt.close();

                print("Train Original Images");
                plt.figure(figsize=(n, n));
                plt.imshow(train_original, origin="upper",interpolation='nearest', cmap="gray");
                plt.savefig(train_output_directory+'orig-img-'+str(epoch)+'.png');
                plt.close();

                print("Train Reconstructed Images");
                plt.figure(figsize=(n, n));
                plt.imshow(train_reconstructed, origin="upper",interpolation='nearest', cmap="gray");
                plt.savefig(train_output_directory+'recons-img-'+str(epoch)+'.png');
                plt.close();

                print("Generated Images");
                plt.figure(figsize=(n1, n1));
                plt.imshow(generated_images, origin="upper",interpolation='nearest', cmap="gray");
                plt.savefig('op/gen-img-'+str(epoch)+'.png');
                plt.close();

            if(epoch%5==0):

                save_path = saver.save(sess, model_directory+'/model_'+str(epoch));
                print("At epoch #",epoch," Model is saved at path: ",save_path);

            print('------------------------------------');
            print('=== Epoch #',epoch,' completed! ===');
            print('------------------------------------');


A = tf.constant(-1.0,shape=[32,32,3]);
A = tf.image.resize_image_with_crop_or_pad(A,33,33);
B = tf.constant(1.0,shape=[33,33,3]);
C = A+B;

def generateImages(model_number):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());
        #saver = tf.train.Saver();

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES);
        saver = tf.train.Saver(var_list=params);

        '''
        for var in params:
            print (var.name+"\t");
        '''
        string = model_directory+'/model_'+str(model_number); 

        try:
            saver.restore(sess, string);
        except:
            print("Previous weights not found of decoder"); 
            sys.exit(0);

        print ("Model loaded");

        batch_size = opts['batch_size'];
        X_test = load_test_data_for_inference();
        X_test = normalize_image(X_test);

        batch_X = get_sequential_cifar_batch(X_test,batch_size);
        
        n1 = 8;
        n_gen = 64;
        sample = tf.random_normal([n_gen,z_dim]);
        generations = sess.run(x_hat_test_padded,feed_dict={z_hat_test:sample.eval()});
        generations = np.reshape(generations,[-1,33,33,3]);
        temp_index = -1;
        for s in range(generations.shape[0]):
            temp_index += 1;
            generations[temp_index] = denormalize_image(generations[temp_index]);

        _C = sess.run(C);

        n1 = 8;
        generated_images = np.empty((33*n1,33*n1,3));
        for i in range(n1):
            for j in range(n1):
                generated_images[i * 33:(i + 1) * 33, j * 33:(j + 1) * 33,:] = _C+generations[i*n1+j];

        print("Generated Images");
        plt.figure(figsize=(n1, n1));
        plt.axis('off');
        plt.imshow(generated_images, origin="upper",interpolation='nearest', cmap="gray");
        plt.savefig('gen-img_1.png');
        plt.close();


def sampleGeneratedImages(model_dir,model_num,gen_output_directory='alpha_gan_inference/',num_images=10000,img_per_batch=1):
    n = img_per_batch;

    if not os.path.exists(gen_output_directory):
        os.makedirs(gen_output_directory);

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES);
        saver = tf.train.Saver(var_list=params);

        #model_directory = 'myVAE-GAN-model_dir_save_when_62_epoch_is_running';
        string = model_directory+'/model_'+str(model_num); 

        try:
            saver.restore(sess, string);
        except:
            print("Previous weights not found of decoder"); 
            sys.exit(0);

        print ("Model loaded successfully from ",string);
        n_batches = int(1.0*num_images/n);
        print('Total n_batches : ',n_batches);
        start_image_number = 0;
        for batch in range(n_batches):
            #start_image_number = n * batch;
            n_gen = n;
            sample = tf.random_normal([n_gen,z_dim]);

            generations = sess.run(x_hat_test,feed_dict={z_hat_test:sample.eval()});
            generations = np.reshape(generations,[-1,32,32,3]);
            temp_index = -1;
            for s in range(generations.shape[0]):
                temp_index += 1;
                generations[temp_index] = denormalize_image(generations[temp_index]);
            for i in range(generations.shape[0]):
                plt.figure(figsize=(0.32, 0.32))
                plt.axis('off');
                plt.imshow(generations[i], origin="upper",interpolation='nearest', cmap="gray",aspect='auto');
                plt.savefig(gen_output_directory+str(start_image_number+1).zfill(6)+'.jpg');
                start_image_number += 1;
                plt.close();
            if(batch%20==0):
                print("Batch #",batch," done !!");

def generateReconstructedImages(num_images,orig_output_directory,recons_output_directory,model_number=20,n=1):
    
    if not os.path.exists(orig_output_directory):
        os.makedirs(orig_output_directory);

    if not os.path.exists(recons_output_directory):
        os.makedirs(recons_output_directory);

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES);
        saver = tf.train.Saver(var_list=params);

        string = model_directory+'/model_'+str(model_number); 

        try:
            saver.restore(sess, string);
        except:
            print("Previous weights not found of decoder"); 
            sys.exit(0);

        print ("Model loaded successfully from ",string);

        n_batches = int(1.0*num_images/n);
        X_test = load_test_data_for_inference();
        X_test = normalize_image(X_test);

        for batch in range(n_batches):
            start_image_number = batch*n;
            stop_image_number = start_image_number + n;
            #print('start_image_number : ',start_image_number);
            #print('stop_image_number : ',stop_image_number);
            batch_X = get_advanced_sequential_cifar_batch(X_test,start_image_number,stop_image_number);
            recons = sess.run(x_hat_test,feed_dict={X:batch_X});
            recons = np.reshape(recons,[-1,32,32,3]);
            temp_index = -1;
            for s in range(batch_X.shape[0]):
                temp_index += 1;
                batch_X[temp_index] = denormalize_image(batch_X[temp_index]);

            temp_index = -1;
            for s in range(recons.shape[0]):
                temp_index += 1;
                recons[temp_index] = denormalize_image(recons[temp_index]);

            _start_image_number = start_image_number;
            for i in range(recons.shape[0]):
                #if i%50==0:
                #    print('Generated image #',i);
                plt.figure(figsize=(0.32, 0.32))
                plt.axis('off');
                plt.imshow(recons[i], origin="upper",interpolation='nearest', cmap="gray",aspect='auto');
                plt.savefig(recons_output_directory+str(_start_image_number+1).zfill(6)+'.jpg');
                _start_image_number += 1;
                plt.close();

            _start_image_number = start_image_number;
            for i in range(batch_X.shape[0]):
                #if i%50==0:
                #    print('Generated image #',i);
                plt.figure(figsize=(0.32, 0.32))
                plt.axis('off');
                plt.imshow(batch_X[i], origin="upper",interpolation='nearest', cmap="gray",aspect='auto');
                plt.savefig(orig_output_directory+str(start_image_number+1).zfill(6)+'.jpg');
                start_image_number += 1;
                plt.close();

            if(batch%20==0):
                print("Batch #",batch," done !!");


def generateImagesSequentially(model_number):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES);
        saver = tf.train.Saver(var_list=params);

        string = model_directory+'/model_'+str(model_number); 

        try:
            saver.restore(sess, string);
        except:
            print("Previous weights not found of decoder"); 
            sys.exit(0);

        print ("Model loaded successfully");
        
        X_test = load_test_data_for_inference();
        X_test = normalize_image(X_test);

        batch_X = get_sequential_cifar_batch(X_test,batch_size);
        
        recons = sess.run(x_hat_test,feed_dict={X:batch_X});
        recons = np.reshape(recons,[-1,32,32,3]);

        n_gen = 100;
        sample = tf.random_normal([n_gen,z_dim]);
        generations = sess.run(x_hat_test,feed_dict={z_hat_test:sample.eval()});
        generations = np.reshape(generations,[-1,32,32,3]);

        temp_index = -1;
        for s in range(generations.shape[0]):
            temp_index += 1;
            generations[temp_index] = denormalize_image(generations[temp_index]);

        temp_index = -1;
        for s in range(batch_X.shape[0]):
            temp_index += 1;
            batch_X[temp_index] = denormalize_image(batch_X[temp_index]);

        temp_index = -1;
        for s in range(recons.shape[0]):
            temp_index += 1;
            recons[temp_index] = denormalize_image(recons[temp_index]);

        n = 5;
        reconstructed = np.empty((32*n,32*n,3));
        original = np.empty((32*n,32*n,3));
        
        for i in range(n):
            for j in range(n):
                original[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32,:] = batch_X[i*n+j];#.reshape([32, 32,3]);
                reconstructed[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32,:] = recons[i*n+j];
                #generated_images[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32,:] = generations[i*n+j];

        n1 = 10;
        generated_images = np.empty((32*n1,32*n1,3));
        for i in range(n1):
            for j in range(n1):
                generated_images[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32,:] = generations[i*n1+j];

        print("Original Images");
        plt.figure(figsize=(n, n));
        plt.imshow(original, origin="upper",interpolation='nearest', cmap="gray");
        plt.savefig('orig-img-model-'+str(model_number)+'.png');
        plt.close();

        print("Reconstructed Images");
        plt.figure(figsize=(n, n));
        plt.imshow(reconstructed, origin="upper",interpolation='nearest', cmap="gray");
        plt.savefig('recons-img-model-'+str(model_number)+'.png');
        plt.close();

        print("Generated Images");
        plt.figure(figsize=(n1, n1));
        plt.imshow(generated_images, origin="upper",interpolation='nearest', cmap="gray");
        plt.savefig('gen-img-model-'+str(model_number)+'.png');
        plt.close();


model_number = 20;
generateImagesSequentially(model_number);





#generateReconstructedImages(num_images=10000,orig_output_directory='original_img/',recons_output_directory='recons_img/',model_number=model_number,n=100);

# num_sets = 5;
# for i in range(1,num_sets+1):
#     print('-'*80);
#     print("Working on set ",i);
#     print('-'*80);
#     sampleGeneratedImages(model_dir=model_directory+"/",model_num=model_number,num_images=10000,img_per_batch=100,gen_output_directory='alpha_gan_generations'+str(i)+"/");

#generateImages(model_number=model_number);
#train();



