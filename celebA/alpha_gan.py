import tensorflow as tf 
import numpy as np 
import os
import sys
import matplotlib.pyplot as plt 
import config
import random,PIL
from PIL import Image
from scipy import misc

#from cifarIO import *

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('tmp/data/',one_hot=True);

tf.set_random_seed(0);
random.seed(0);
'''
For better readability
I will be using same naming conventions as provided in
pseudo code of official paper.
'''

#Used to initialize kernel weights
stddev = 0.01;#99999;

opts = config.config_celebA
crop_style = opts['celebA_crop'];
celeb_source = opts['dataset_dir'];

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

#Given absolute image path, returns image in numpy
def read_image(path): 
    logging.debug('In read_image function for path : ',path);
    img = misc.imread(path);
    return img;

#Returns meta data about image i.e height,width,num_channels
def image_meta(image):
    logging.debug('In image_meta function');
    return image.shape[0],image.shape[1],image.shape[2];
    #return image.size[0],image.size[1],opts['num_channels'];

def denormalize_image(image):
    image /= 2. #rescale value from [-1,1] to [-0.5,0.5]
    image = image + 0.5 #rescale value from [-0.5,0.5] to [0,1]
    return image;

def plot_denormalized_image(image,title):
    image = denormalize_image(image);
    plt.figure();
    plt.title(title);
    plt.imshow(image);

def plot_image(image,title):
    plt.figure();
    plt.axis('off');
    plt.title(title);
    plt.imshow(image);
    
    #plt.show();
    #plt.close();

def crop(im):
    width = 178
    height = 218
    new_width = 140
    new_height = 140
    if crop_style == 'closecrop':
        # This method was used in DCGAN, pytorch-gan-collection, AVB, ...
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height)/2
        im = im.crop((left, top, right, bottom))
        im = im.resize((64, 64), PIL.Image.ANTIALIAS)
    elif self.crop_style == 'resizecrop':
        # This method was used in ALI, AGE, ...
        im = im.resize((64, 78), PIL.Image.ANTIALIAS)
        im = im.crop((0, 7, 64, 64 + 7))
    else:
        raise Exception('Unknown crop style specified')
    return np.array(im).reshape(64, 64, 3) / 255.

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def normalize_image(image):
    normalized_image = image - 0.5;
    normalized_image *= 2;
    # normalized_image = (image - np.min(image))/(np.max(image)-np.min(image));
    # normalized_image = 2*normalized_image;
    # normalized_image -= 1;
    return normalized_image;

def get_random_batch(file_iter,batch_size = 3,):
    random_file_iter = np.random.choice(file_iter,batch_size,replace=False);
    #print("Random_file_iter");
    #print(random_file_iter);
    #sys.exit(0);
    X = np.zeros([len(random_file_iter),opts['img_height'],opts['img_width'],opts['num_channels']]);
    #print(X.shape);
    index = -1;
    for f in random_file_iter:
        index += 1;
        f = f + '.jpg';
        curr_img = Image.open(f);
        curr_img = crop(curr_img);
        #print(np.min(curr_img));
        curr_img = normalize_image(curr_img);
        #print(np.min(curr_img));
        #print (curr_img.shape);
        X[index] = curr_img;
    return X;



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

        #64x64x3 --> means size of input before applying conv1
        conv1 = tf.layers.conv2d(X,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=64,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv1_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv1 = tf.layers.batch_normalization(conv1,training=isTrainable,reuse=reuse,name='bn_1');
        conv1 = tf.nn.relu(conv1,name='leaky_relu_conv_1');
        
        #32x32x64
        conv2 = tf.layers.conv2d(conv1,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=128,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv2_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv2 = tf.layers.batch_normalization(conv2,training=isTrainable,reuse=reuse,name='bn_2');
        conv2 = tf.nn.relu(conv2,name='leaky_relu_conv_2');
        
        #16x16x128
        conv3 = tf.layers.conv2d(conv2,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=256,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv3_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv3 = tf.layers.batch_normalization(conv3,training=isTrainable,reuse=reuse,name='bn_3');
        conv3 = tf.nn.relu(conv3,name='leaky_relu_conv_3');
    
        #8x8x256
        conv4 = tf.layers.conv2d(conv3,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=512,kernel_size=[5,5],padding='SAME',strides=(1,1),name='enc_conv4_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv4 = tf.layers.batch_normalization(conv4,training=isTrainable,reuse=reuse,name='bn_4');
        conv4 = tf.nn.relu(conv4,name='leaky_relu_conv_4');
        #8x8x512
        conv5 = tf.layers.conv2d(conv4,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=1024,kernel_size=[3,3],padding='SAME',strides=(2,2),name='enc_conv5_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv5 = tf.layers.batch_normalization(conv5,training=isTrainable,reuse=reuse,name='bn_5');
        conv5 = tf.nn.relu(conv5,name='leaky_relu_conv_5');        
        
        #4x4x1024
        conv5_flattened = tf.layers.flatten(conv5);
        latent_code = tf.layers.dense(conv5_flattened,z_dim,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),activation=None,name='enc_latent_space',trainable=isTrainable,reuse=reuse);
        return latent_code;

def generator(z_sample,isTrainable=True,reuse=False,name='theta_generator'):
    with tf.variable_scope(name) as scope:  
        #decoder_activations = {};
        if reuse:
            scope.reuse_variables();
        #print('z_sample.shape: ',z_sample.shape);
        z_sample = tf.layers.dense(z_sample,4*4*1024,activation=None,trainable=isTrainable,reuse=reuse,name='dec_dense_fc_first_layer',kernel_initializer=tf.truncated_normal_initializer(stddev=stddev));
        z_sample = tf.layers.batch_normalization(z_sample,training=isTrainable,reuse=reuse,name='bn_0');
        z_sample = tf.nn.relu(z_sample);
        z_sample = tf.reshape(z_sample,[-1,4,4,1024]);
        #8x8x512

        deconv1 = tf.layers.conv2d_transpose(z_sample,kernel_initializer=tf.random_normal_initializer(stddev=stddev),filters=512,kernel_size=[3,3],padding='SAME',activation=None,strides=(2,2),name='dec_deconv1_layer',trainable=isTrainable,reuse=reuse); # 16x16
        deconv1 = tf.layers.batch_normalization(deconv1,training=isTrainable,reuse=reuse,name='bn_1');
        deconv1 = tf.nn.relu(deconv1,name='relu_deconv_1');
         
        # #16x16x256
        deconv2 = tf.layers.conv2d_transpose(deconv1,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=256,kernel_size=[5,5],padding='SAME',activation=None,strides=(2,2),name='dec_deconv2_layer',trainable=isTrainable,reuse=reuse); # 16x16
        deconv2 = tf.layers.batch_normalization(deconv2,training=isTrainable,reuse=reuse,name='bn_2');
        deconv2 = tf.nn.relu(deconv2,name='relu_deconv_2');
        
        #32x32x128
        deconv3 = tf.layers.conv2d_transpose(deconv2,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=128,kernel_size=[5,5],padding='SAME',activation=None,strides=(2,2),name='dec_deconv3_layer',trainable=isTrainable,reuse=reuse); # 16x16
        deconv3 = tf.layers.batch_normalization(deconv3,training=isTrainable,reuse=reuse,name='bn_3');
        deconv3 = tf.nn.relu(deconv3,name='relu_deconv_3');

        deconv4 = tf.layers.conv2d_transpose(deconv3,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=64,kernel_size=[5,5],padding='SAME',activation=None,strides=(2,2),name='dec_deconv4_layer',trainable=isTrainable,reuse=reuse); # 16x16
        deconv4 = tf.layers.batch_normalization(deconv4,training=isTrainable,reuse=reuse,name='bn_4');
        deconv4 = tf.nn.relu(deconv4,name='relu_deconv_4');
        
        #64x64x64 
        deconv5 = tf.layers.conv2d_transpose(deconv4,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=3,kernel_size=[5,5],padding='SAME',activation=None,strides=(1,1),name='dec_deconv5_layer',trainable=isTrainable,reuse=reuse); # 16x16    
        #deconv4 = tf.layers.dropout(deconv4,rate=keep_prob,training=True);
        deconv5 = tf.nn.tanh(deconv5);
        #64x64x3
        
        deconv_5_reshaped = tf.reshape(deconv5,[-1,img_height,img_width,num_channels]);
        return deconv_5_reshaped;


def discriminator(X,isTrainable=True,reuse=False,name='phi_discriminator'):
    with tf.variable_scope(name) as scope:

        if reuse:
            scope.reuse_variables();

        #64x64x3 --> means size of input before applying conv1
        conv1 = tf.layers.conv2d(X,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=64,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv1_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv1 = tf.layers.batch_normalization(conv1,training=isTrainable,reuse=reuse,name='bn_1');
        conv1 = tf.nn.relu(conv1,name='leaky_relu_conv_1');
        
        #32x32x64
        conv2 = tf.layers.conv2d(conv1,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=128,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv2_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv2 = tf.layers.batch_normalization(conv2,training=isTrainable,reuse=reuse,name='bn_2');
        conv2 = tf.nn.relu(conv2,name='leaky_relu_conv_2');
        
        #16x16x128
        conv3 = tf.layers.conv2d(conv2,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=256,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv3_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv3 = tf.layers.batch_normalization(conv3,training=isTrainable,reuse=reuse,name='bn_3');
        conv3 = tf.nn.relu(conv3,name='leaky_relu_conv_3');
    
        #8x8x256
        conv4 = tf.layers.conv2d(conv3,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=512,kernel_size=[5,5],padding='SAME',strides=(1,1),name='enc_conv4_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv4 = tf.layers.batch_normalization(conv4,training=isTrainable,reuse=reuse,name='bn_4');
        conv4 = tf.nn.relu(conv4,name='leaky_relu_conv_4');

        conv5 = tf.layers.conv2d(conv4,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=1024,kernel_size=[3,3],padding='SAME',strides=(2,2),name='enc_conv5_layer',activation=None,trainable=isTrainable,reuse=reuse); 
        conv5 = tf.layers.batch_normalization(conv5,training=isTrainable,reuse=reuse,name='bn_5');
        conv5 = tf.nn.relu(conv5,name='leaky_relu_conv_5'); 
        
        conv5_flattened = tf.layers.flatten(conv5);
        output_disc = tf.layers.dense(conv5_flattened,1,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),activation=None,name='dis_fc_layer',trainable=isTrainable,reuse=reuse);
        
        return output_disc;

def code_discriminator(Z,isTrainable=True,reuse=False,name='omega_code_discriminator'):
    with tf.variable_scope(name) as scope:

        if reuse:
            scope.reuse_variables();

        #kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),S

        fc1 = tf.layers.dense(Z,750,activation=None,name='code_dis_fc_layer_1',trainable=isTrainable,reuse=reuse);
        #fc1 = tf.layers.batch_normalization(fc1,training=isTrainable,reuse=reuse,name='bn_1');
        fc1 = tf.nn.leaky_relu(fc1);

        fc2 = tf.layers.dense(fc1,750,activation=None,name='code_dis_fc_layer_2',trainable=isTrainable,reuse=reuse);
        #fc2 = tf.layers.batch_normalization(fc2,training=isTrainable,reuse=reuse,name='bn_2');
        fc2 = tf.nn.leaky_relu(fc2);

        fc3 = tf.layers.dense(fc2,750,activation=None,name='code_dis_fc_layer_3',trainable=isTrainable,reuse=reuse);
        #fc3 = tf.layers.batch_normalization(fc3,training=isTrainable,reuse=reuse,name='bn_3');
        fc3 = tf.nn.leaky_relu(fc3);

        logits = tf.layers.dense(fc3,1,activation=None,name='code_dis_fc_layer_4',trainable=isTrainable,reuse=reuse);

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

x_prime = generator(z_prime,reuse=True);

z_hat_logits = code_discriminator(z_hat);
z_prime_logits = code_discriminator(z_prime,reuse=True);

x_hat_logits = discriminator(x_hat);
x_prime_logits = discriminator(x_prime,reuse=True);
x_logits = discriminator(X,reuse=True);

lamda_enc = opts['lamda_enc'];
lamda_gen = opts['lamda_gen'];
l1_recons_loss = tf.reduce_mean(tf.abs(X - x_hat));

encoder_loss = lamda_enc*l1_recons_loss + RC_w(z_hat_logits);

generator_loss = lamda_gen*l1_recons_loss + RD_phi(x_hat_logits) + RD_phi(x_prime_logits);

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
    encoder_optimizer = tf.train.AdamOptimizer(learning_rate=encoder_learning_rate,beta1=0.5);
    encoder_gradsVars = encoder_optimizer.compute_gradients(encoder_loss, eta_encoder_params);
    encoder_train_optimizer = encoder_optimizer.apply_gradients(encoder_gradsVars);

    generator_optimizer = tf.train.AdamOptimizer(learning_rate=generator_learning_rate,beta1=0.5);
    generator_gradsVars = generator_optimizer.compute_gradients(generator_loss, theta_generator_params);
    generator_train_optimizer = generator_optimizer.apply_gradients(generator_gradsVars);

    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=discriminator_learning_rate,beta1=0.5);
    discriminator_gradsVars = discriminator_optimizer.compute_gradients(discriminator_loss, phi_discriminator_params);
    discriminator_train_optimizer = discriminator_optimizer.apply_gradients(discriminator_gradsVars);

    code_discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=code_discriminator_learning_rate,beta1=0.5);
    code_discriminator_gradsVars = code_discriminator_optimizer.compute_gradients(code_discriminator_loss, omega_code_discriminator_params);
    code_discriminator_train_optimizer = code_discriminator_optimizer.apply_gradients(code_discriminator_gradsVars);

tf.summary.scalar("encoder_loss",encoder_loss);
tf.summary.scalar("discriminator_loss",discriminator_loss);
tf.summary.scalar("generator_loss",generator_loss);
tf.summary.scalar("code_discriminator_loss",code_discriminator_loss);
tf.summary.scalar("Unscaled l1_recons_loss",l1_recons_loss);
tf.summary.scalar("encoder l1_recons_loss",lamda_enc*l1_recons_loss);
tf.summary.scalar("generator l1_recons_loss",lamda_gen*l1_recons_loss);

all_gradsVars = [encoder_gradsVars, generator_gradsVars, discriminator_gradsVars, code_discriminator_gradsVars];

for grad_vars in all_gradsVars:
    for g,v in grad_vars:  
        tf.summary.histogram(v.name,v)
        tf.summary.histogram(v.name+str('grad'),g)

merged_all = tf.summary.merge_all();
log_directory = './myAlpha-GAN-dir';
model_directory='./myAlpha-GAN-model_dir';
output_directory = './op/';
train_output_directory = './train-op/';

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

    # X_train = load_data();
    # X_test = load_test_data();
    
    # n_batches = X_train.shape[0]/batch_size;#mnist.train.num_examples/batch_size;
    # n_batches = int(n_batches);

    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer());

        ###########################
        #DATA READING
        ###########################
        mode = 'train';
        train_min_file_num = opts['train_min_filenum'];
        train_max_file_num = opts['train_max_filenum'];
        train_files = range(train_min_file_num, 1+train_max_file_num);
        train_file_iter=[os.path.join(celeb_source, '%s' % str(i).zfill(6)) for i in train_files]

        val_min_file_num = opts['val_min_filenum'];
        val_max_file_num = opts['val_max_filenum'];
        val_files = range(val_min_file_num, 1+val_max_file_num);
        val_file_iter=[os.path.join(celeb_source, '%s' % str(i).zfill(6)) for i in val_files]
        n_batches = 162770/batch_size;#mnist.train.num_examples/batch_size;
        n_batches = int(n_batches);
        
        print('-'*80);
        print('n_batches : ',n_batches,' when batch_size: ',batch_size);
        #for tensorboard
        saver = tf.train.Saver(max_to_keep=5);
        writer = tf.summary.FileWriter(log_directory,sess.graph);
        iterations = 0;
        
        for epoch in range(n_epoch):
            for batch in range(n_batches):
                iterations += 1;
                
                #Train Encoder
                h=2;
                for i in range(h):
                    X_batch = get_random_batch(train_file_iter,batch_size);
                    fd = {X: X_batch};
                    _,enc_loss= sess.run([encoder_train_optimizer,encoder_loss],feed_dict = fd);

                #Train Generator
                j=2;
                for i in range(j):
                    X_batch = get_random_batch(train_file_iter,batch_size);
                    fd = {X: X_batch};
                    _,gen_loss= sess.run([generator_train_optimizer,generator_loss],feed_dict = fd);

                #Train Discriminator
                k=1;
                for i in range(k):
                    X_batch = get_random_batch(train_file_iter,batch_size);
                    fd = {X: X_batch};
                    _,disc_loss= sess.run([discriminator_train_optimizer,discriminator_loss],feed_dict = fd);

                #Train Code Discriminator
                l=1;
                for i in range(l):
                    X_batch = get_random_batch(train_file_iter,batch_size);
                    fd = {X: X_batch};
                    _,code_disc_loss,merged= sess.run([code_discriminator_train_optimizer,code_discriminator_loss,merged_all],feed_dict = fd);

                if(iterations%20==0):
                    writer.add_summary(merged,iterations);

                if(batch%200 == 0):
                    print('Batch #',batch,' done!');

                #break;

            if(epoch%2==0):

                num_val_img = 25;
                batch_X = get_random_batch(val_file_iter,num_val_img);
                
                recons = sess.run(x_hat_test,feed_dict={X:batch_X});
                recons = np.reshape(recons,[-1,64,64,3]);


                train_batch_X = get_random_batch(train_file_iter,num_val_img);
                train_recons = sess.run(x_hat_test,feed_dict={X:train_batch_X});
                train_recons = np.reshape(train_recons,[-1,64,64,3]);

                n_gen = 100;
                sample = tf.random_normal([n_gen,z_dim]);
                generations = sess.run(x_hat_test,feed_dict={z_hat_test:sample.eval()});
                generations = np.reshape(generations,[-1,64,64,3]);

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
                reconstructed = np.empty((64*n,64*n,3));
                original = np.empty((64*n,64*n,3));
                
                for i in range(n):
                    for j in range(n):
                        original[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64,:] = batch_X[i*n+j];#.reshape([64, 64,3]);
                        reconstructed[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64,:] = recons[i*n+j];
                        #generated_images[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64,:] = generations[i*n+j];

                n = 5;
                train_reconstructed = np.empty((64*n,64*n,3));
                train_original = np.empty((64*n,64*n,3));
                
                for i in range(n):
                    for j in range(n):
                        train_original[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64,:] = train_batch_X[i*n+j];#.reshape([64, 64,3]);
                        train_reconstructed[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64,:] = train_recons[i*n+j];

                n1 = 10;
                generated_images = np.empty((64*n1,64*n1,3));
                for i in range(n1):
                    for j in range(n1):
                        generated_images[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64,:] = generations[i*n1+j];

                print("Original Images");
                plt.figure(figsize=(n, n));
                plt.imshow(original, origin="upper",interpolation='nearest', cmap="gray");
                plt.savefig(output_directory+'orig-img-'+str(epoch)+'.png');
                plt.close();

                print("Reconstructed Images");
                plt.figure(figsize=(n, n));
                plt.imshow(reconstructed, origin="upper",interpolation='nearest', cmap="gray");
                plt.savefig(output_directory+'recons-img-'+str(epoch)+'.png');
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
                plt.savefig(output_directory+'gen-img-'+str(epoch)+'.png');
                plt.close();

            if(epoch%5==0):

                save_path = saver.save(sess, model_directory+'/model_'+str(epoch));
                print("At epoch #",epoch," Model is saved at path: ",save_path);

            print('------------------------------------');
            print('=== Epoch #',epoch,' completed! ===');
            print('------------------------------------');


train();


