"""
Created on Sun Nov 18 12:06:09 2018

@author: linkermann

improved Wasserstein GAN simplified
"""
import os, sys
sys.path.append(os.getcwd())
import tflib.save_images as imsaver
import tflib.plot as plotter
import tflib.mnist as mnistloader
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as lays


# --- parameter ------------------------------------------------------------------------------
EXPERIMENT = "vae_10"  # path to results folder to save/restore
MODE = 'vae' # choose between 'plain', 'cond', 'cond_ordered', 'enc', 'vae'
#'cond' includes conditional input (labels) to G and D [conditional GAN], output input ims compared to generated
#'cond_ordered' includes conditional input (labels) to G and D [conditional GAN], output 5 of each digit
# 'enc' includes Encoder before G (with mean&var sampling) as well as conditional input(labels) to D
# 'vae' comprises Encoder and Generator (with mean&var sampling), but no Discriminator, reconstructs test image for output & samples from N(0,I) to generate new output

BATCH_SIZE = 50 # Batch size needs to divide 50000
LEARNING_RATE = 0.001 # 0.0002 # Adam learning rate for GAN (both G and D)
DIM = 10 # 64 # model dimensionality for both Generator and Discriminator
NOISE_DIM = 10 # 60# noise input dim for Generator (plain and cond)
Z_DIM = 10 # 60 for enc/vae: output dim of Encoder, input dim for Generator
LAMBDA = 10 # wgan-gp: Gradient penalty lambda hyperparameter
DISC_ITERS = 5 # wgan-gp: How many discriminator iterations per generator iteration
ITERS = 5000 # How many generator iterations to train for
START_ITER = 0  # Default 0 (start new learning), set accordingly if restoring from checkpoint (100, 200, ...)

IM_DIM = 28 # number of pixels along x and y (square assumed) # mnist: 28
output_dim = IM_DIM*IM_DIM # Number of pixels (mnist: 28*28*1) 
N_LABELS = 10
restore_path = "/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/results/" + EXPERIMENT + "/model.ckpt" # desktop path
log_dir = "/tmp/tensorflow_mnist" # the path of tensorflow's log

# Start tensorboard in the current folder:   tensorboard --logdir=logdir
# Open 'Webbrowser at http://localhost:6006/
 

# ---------------- help functions ---------------------------------------------
def get_shape(tensor): 		              # prints the shape of any tensor
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims

def print_current_model_settings(locals_):    # prints the chosen parameter
    all_vars = [(k,v) for (k,v) in locals_.items() if k.isupper()] 
    all_vars = sorted(all_vars, key=lambda x: x[0])
    for var_name, var_value in all_vars:
        print("\t{}: {}".format(var_name, var_value))
    
# ---------------- network parts ----------------------------------------

def Encoder(inputs): 
    inputs = tf.reshape(inputs, [BATCH_SIZE, 1, IM_DIM, IM_DIM]) 
    out = lays.conv2d(inputs, DIM, kernel_size = 5, stride = 2, # DIM = 64
        data_format='NCHW', activation_fn=tf.nn.leaky_relu, scope='Enc.1',
        weights_initializer=tf.initializers.he_uniform(), reuse=tf.AUTO_REUSE)
    out = lays.conv2d(out, 2*DIM, kernel_size = 5, stride = 2, #2*DIM = 128
        data_format='NCHW', activation_fn=tf.nn.leaky_relu, scope='Enc.2',
        weights_initializer=tf.initializers.he_uniform(), reuse=tf.AUTO_REUSE)
    out = lays.conv2d(out, 4*DIM, kernel_size = 5, stride = 2, # 4*DIM = 256
        data_format='NCHW', activation_fn=tf.nn.leaky_relu, scope='Enc.3',
        weights_initializer=tf.initializers.he_uniform(), reuse=tf.AUTO_REUSE)
    out = tf.reshape(out, [-1, 4*4*4*DIM]) # adjust
    encoded = lays.fully_connected(out, Z_DIM, activation_fn=None, scope='Enc.Code',
        weights_initializer=tf.initializers.glorot_uniform(), reuse=tf.AUTO_REUSE)
    mean = lays.fully_connected(out, Z_DIM, activation_fn=None, scope='Enc.Mean',
        weights_initializer=tf.initializers.glorot_uniform(), reuse=tf.AUTO_REUSE)
    variance = lays.fully_connected(out, Z_DIM, activation_fn=None, scope='Enc.Var',
        weights_initializer=tf.initializers.glorot_uniform(), reuse=tf.AUTO_REUSE)
    return encoded, mean, variance  # shape of all: [BATCH_SIZE, Z_DIM])

def sample_z(shape, mean, variance): # variance has to be positive
    stddev = tf.exp(0.5 * variance) # standard deviation = square root of variance 
    noise = tf.random_normal(shape, mean=0., stddev=1.) # sample from N(0,1)
    return (noise * stddev) + mean  # change scale & location  

def Generator(n_samples, conditions=None, mean=None, variance=None, noise=None):  
    if noise is None:          # sample from Gaussian        
        noise = tf.random_normal([n_samples, NOISE_DIM], mean= 0.0, stddev = 1.0) 
    if ((MODE == 'enc' or MODE == 'vae') and mean != None and variance != None): # input: mean and var
        inputs = sample_z([n_samples, Z_DIM], mean, variance)
    elif (MODE == 'cond' or MODE == 'cond_ordered'):       # input: labels
        labels = tf.one_hot(tf.cast(conditions, tf.uint8), 10) # [BATCH_SIZE, 10]
        inputs = tf.concat([noise, labels], 1) #to: (BATCH_SIZE, NOISE_DIM+10) 
    else:
        inputs = noise # (BATCH_SIZE, NOISE_DIM)             
    out = lays.fully_connected(inputs, 4*4*4*DIM, reuse = tf.AUTO_REUSE, # expansion
        weights_initializer=tf.initializers.glorot_uniform(), scope = 'Gen.Input')
    out = tf.reshape(out, [-1, 4*DIM, 4, 4])
    out = tf.transpose(out, [0,2,3,1], name='NCHW_to_NHWC')
    out = lays.conv2d_transpose(out, 2*DIM, kernel_size=5, stride=2, scope='Gen.1',
        weights_initializer=tf.initializers.he_uniform(), reuse=tf.AUTO_REUSE)
    out = out[:,:7,:7,:]  # because output needs to be 28x28
    out = lays.conv2d_transpose(out, DIM, kernel_size=5, stride=2, scope='Gen.2',
        weights_initializer=tf.initializers.he_uniform(), reuse=tf.AUTO_REUSE)
    out = lays.conv2d_transpose(out, 1, kernel_size=5, stride=2, scope='Gen.3',
        weights_initializer=tf.initializers.he_uniform(), reuse=tf.AUTO_REUSE,
        activation_fn=tf.nn.sigmoid)
    out = tf.transpose(out, [0,3,1,2], name='NHWC_to_NCHW')
    return tf.reshape(out, [BATCH_SIZE, output_dim])

def Discriminator(inputs, conditions=None):
    if (MODE == 'plain'):               # inputs: [BATCH_SIZE, output_dim]
        ins = tf.reshape(inputs, [BATCH_SIZE, 1, IM_DIM, IM_DIM]) 
    else:                               # input: labels [BATCH_SIZE, 1]         
        labels = tf.one_hot(tf.cast(conditions, tf.uint8), 10) # [BATCH_SIZE, 10]
        pad = tf.tile(labels, [1, 24]) # expand to pad image: [BATCH_SIZE, 240]
        ins = tf.concat([inputs, pad], 1) #to: (BATCH_SIZE, 1*32*32) 
        ins = tf.reshape(ins, [BATCH_SIZE, 1, 32, 32]) 
    out = lays.conv2d(ins, DIM, kernel_size=5, stride=2, reuse=tf.AUTO_REUSE,
        data_format='NCHW', activation_fn=tf.nn.leaky_relu, 
        weights_initializer=tf.initializers.he_uniform(), scope='Disc.1')
    out = lays.conv2d(out, 2*DIM, kernel_size=5, stride=2, reuse=tf.AUTO_REUSE,
        data_format='NCHW', activation_fn=tf.nn.leaky_relu, 
        weights_initializer=tf.initializers.he_uniform(), scope='Disc.2')
    out = lays.conv2d(out, 4*DIM, kernel_size=5, stride=2, reuse=tf.AUTO_REUSE,
        data_format='NCHW', activation_fn=tf.nn.leaky_relu, 
        weights_initializer=tf.initializers.he_uniform(), scope='Disc.3')
    out = tf.reshape(out, [-1, 4*4*4*DIM]) # adjust   
    out = lays.fully_connected(out, 1, activation_fn=None, reuse = tf.AUTO_REUSE,   # to single value
        weights_initializer=tf.initializers.glorot_uniform(), scope = 'Disc.Out')
    return tf.reshape(out, [-1])


#------------------------- networks pipeline structure ----------------------------------------------------

print_current_model_settings(locals().copy())

tf.reset_default_graph()

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, output_dim]) 

if(MODE == 'enc'):
    condition_data = tf.placeholder(tf.int32, shape=[BATCH_SIZE]) # labels for D
    encoded_data, mean_data, variance_data = Encoder(real_data)
    fake_data = Generator(BATCH_SIZE, mean=mean_data, variance=variance_data)
    disc_real = Discriminator(real_data, conditions=condition_data)
    disc_fake = Discriminator(fake_data, conditions=condition_data)
elif(MODE == 'cond' or MODE == 'cond_ordered'):
    condition_data = tf.placeholder(tf.int32, shape=[BATCH_SIZE]) # labels for G and D
    fake_data = Generator(BATCH_SIZE, conditions=condition_data)
    disc_real = Discriminator(real_data, conditions=condition_data)
    disc_fake = Discriminator(fake_data, conditions=condition_data)
elif(MODE == 'vae'):
    encoded_data, mean_data, variance_data = Encoder(real_data)
    fake_data = Generator(BATCH_SIZE, mean=mean_data, variance=variance_data)
else:
    fake_data = Generator(BATCH_SIZE)
    disc_real = Discriminator(real_data)
    disc_fake = Discriminator(fake_data)

fake_image = tf.reshape(fake_data, [BATCH_SIZE, IM_DIM, IM_DIM, 1])
G_image = tf.summary.image("G_out", fake_image)
if(MODE != 'vae'):
    D_prob_sum = tf.summary.histogram("D_prob", disc_real)
    G_prob_sum = tf.summary.histogram("G_prob", disc_fake)


## Classificator:
x = tf.placeholder(tf.float32, [None, 784], name='x-input') # images flattened
y = tf.placeholder(tf.float32, [None, N_LABELS], name='y-input') # labels in one-hot
out1 = tf.contrib.layers.fully_connected(x, 800, activation_fn=tf.nn.relu) # 2 layer NN with 800 HU
out2 = tf.contrib.layers.fully_connected(out1, N_LABELS, activation_fn=tf.nn.softmax)
cross_entropy = -tf.reduce_mean(y * tf.log(out2))  # loss
train_step = tf.train.GradientDescentOptimizer(10.).minimize(cross_entropy)  # train
correct_prediction = tf.equal(tf.argmax(out2, 1), tf.argmax(y, 1))  # test
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # calculate accuracy


# ------------------- make it a WGAN-GP: loss function and training ops --------------------------------------------
    
if(MODE != 'vae'):    
    # Standard WGAN loss
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    loss_sum = tf.summary.scalar("D_loss", disc_cost)
    G_loss_sum = tf.summary.scalar("G_loss", gen_cost)

    # Gradient penalty on Discriminator 
    alpha = tf.random_uniform(shape=[BATCH_SIZE,1], minval=0., maxval=1.)
    interpolates = real_data + (alpha*(fake_data - real_data))
    if(MODE == 'plain'):
        gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0] 
    else:
        gradients = tf.gradients(Discriminator(interpolates, conditions=condition_data), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    loss_sum_gp = tf.summary.scalar("D_loss_gp", disc_cost)

    merged_summary_op_d = tf.summary.merge([loss_sum, loss_sum_gp, D_prob_sum])
    merged_summary_op_g = tf.summary.merge([G_loss_sum, G_prob_sum, G_image])
else:
    # Standard VAE loss
    ## reconstruction loss: pixel-wise L2 loss = mean(square(gen_image - real_im)) # E[log P(X|z)]
    img_loss = tf.reduce_sum(tf.squared_difference(fake_data, real_data), 1)   # has to be sum, not mean!
    mean_img_loss = tf.reduce_mean(img_loss)
    ## latent loss = KL(latent code, unit gaussian) # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    latent_loss = -0.5 * tf.reduce_mean(1. + variance_data - tf.square(mean_data) - tf.exp(variance_data), 1) # better mean than sum
    mean_latent_loss = tf.reduce_mean(latent_loss)
    vae_loss = tf.reduce_mean(img_loss + latent_loss)  
    img_loss_sum = tf.summary.scalar("img_loss", mean_img_loss)
    latent_loss_sum = tf.summary.scalar("latent_loss", mean_latent_loss)
    vae_loss_sum = tf.summary.scalar("VAE_loss", vae_loss)
    merged_summary_op_vae = tf.summary.merge([vae_loss_sum, img_loss_sum, latent_loss_sum, G_image])

enc_gen_params = [var for var in tf.get_collection("variables") if ("Gen" in var.name or "Enc" in var.name)]  # Encoder trained together with Generator
gen_params = [var for var in tf.get_collection("variables") if "Gen" in var.name] 
disc_params = [var for var in tf.get_collection("variables") if "Disc" in var.name]

# ADAM defaults: learning_rate=0.001, beta1=0.9, beta2=0.999
if(MODE == 'enc'):
    gen_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=enc_gen_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)
elif(MODE == 'vae'): # adjust? LEARNING_RATE = 1e-4 # 0.0002 # 0.01 # 0.0005
    train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(vae_loss, var_list=enc_gen_params) 
else:
    gen_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

init_op = tf.global_variables_initializer()
saver = tf.train.Saver() # ops to save and restore all the variables.


# -------------------- Dataset iterators -------------------------------------------------------------------
train_gen, dev_gen, test_gen = mnistloader.load(BATCH_SIZE, BATCH_SIZE)	# load sets into global vars

def inf_train_gen():
    while True:			# infinite iterator for training
        for images,targets in train_gen():
            yield (images,targets)
            
def inf_dev_gen():
    while True:			# infinite iterator for validation
        for images,targets in dev_gen():
            yield (images,targets)
            
def inf_test_gen():
    while True:			# infinite iterator for testing
        for images,targets in test_gen():
            yield (images,targets)

gen = inf_train_gen()			# init iterator for training set
dev_generator = inf_dev_gen()		# init iterator for validation set 
test_generator = inf_test_gen()		# init iterator for test set 


# --------------------------------- testing: generate samples -----------------------------------------------
fixed_labels_array = np.tile(np.arange(10), 5) # create labels in order (same digit below): 5*10=50 (BATCH_SIZE)
fixed_data, fixed_labels = next(test_generator) # [0,1] # first batch no 8, take second test batch
fixed_data, fixed_labels = next(test_generator) # [0,1]
#sort test batch
indices = np.argsort(fixed_labels) 
sorted_labels = fixed_labels[indices]
sorted_data = fixed_data[indices]

if(START_ITER > 0):    # get noise from saved model: variable, implicit float
    fixed_noise = tf.get_variable("noise", shape=[BATCH_SIZE, NOISE_DIM]) 
else:
    fixed_noise = tf.Variable(tf.random_normal(shape=[BATCH_SIZE, NOISE_DIM]), name='noise') 

if(MODE == 'enc' or MODE == 'vae'):
    fixed_data_255 = ((sorted_data)*255.).astype('int32') # [0,255] 
    fixed_codes, fixed_mean, fixed_variance = Encoder(sorted_data)
    fixed_noise_samples = Generator(BATCH_SIZE, mean=fixed_mean, variance=fixed_variance, noise=fixed_noise)
elif(MODE == 'cond_ordered'):
    fixed_noise_samples = Generator(BATCH_SIZE, conditions=fixed_labels_array, noise=fixed_noise)
elif(MODE == 'cond'):
    fixed_data_255 = ((sorted_data)*255.).astype('int32') # [0,255] 
    fixed_noise_samples = Generator(BATCH_SIZE, conditions=sorted_labels, noise=fixed_noise)
else:
    fixed_noise_samples = Generator(BATCH_SIZE, noise=fixed_noise)
if(MODE == 'vae'):
    # noise_for_vae = tf.random_normal(shape, mean=0., stddev=1.) # sample from N(0,1)
    more_noise_samples = Generator(BATCH_SIZE, noise=fixed_noise)
    
def generate_image(frame, final): # generates a batch of samples next to each other in one image!
    inference_start_time = time.time()  
    if(MODE == 'cond_ordered'):
        samples = session.run(fixed_noise_samples, feed_dict={condition_data: fixed_labels_array}) # [0,1]
    elif(MODE == 'cond'):
        samples = session.run(fixed_noise_samples, feed_dict={condition_data: sorted_labels}) # [0,1]
    elif(MODE == 'plain'):
        samples = session.run(fixed_noise_samples) # [0,1]
    else:
        samples = session.run(fixed_noise_samples, feed_dict={real_data: sorted_data}) # [0,1]
    inference_end_time = time.time()  # inference time analysis
    inference_time = (inference_end_time - inference_start_time)
    print("The architecture took ", inference_time, "sec for the generation of ", BATCH_SIZE, "images")

    samples_255 = ((samples)*(255.)).astype('uint8') # [0,255] 

    if(MODE == 'enc' or MODE == 'vae' or MODE == 'cond'):
        for i in range(0, BATCH_SIZE):
            samples_255 = np.insert(samples_255, i*2, fixed_data_255[i,:], axis=0) # real (cond digit) next to sample
        imsaver.save_images(samples_255.reshape((2*BATCH_SIZE, IM_DIM, IM_DIM)), 'samples_{}.jpg'.format(frame), alternate_viz=True, conds=True)
        print("Iteration %d :" % frame)
        # compare generated to real one
        real = tf.reshape(sorted_data, [BATCH_SIZE,IM_DIM,IM_DIM,1])
        pred = tf.reshape(samples, [BATCH_SIZE,IM_DIM,IM_DIM,1])
        ssim_vals = tf.image.ssim(real, pred, max_val=1.0) # on batch
        mse_vals = tf.reduce_mean(tf.keras.metrics.mse(real, pred), [1,2]) # mse on grayscale, on [0,1]
        psnr_vals = tf.image.psnr(real, pred, max_val=1.0) # on batch, out tf.float32
        ssim_avg = (tf.reduce_mean(ssim_vals)).eval()
        mse_avg = (tf.reduce_mean(mse_vals)).eval()
        psnr_avg = (tf.reduce_mean(psnr_vals)).eval()
        plotter.plot('SSIM avg', ssim_avg) # show average of ssim and mse vals over whole batch
        plotter.plot('MSE avg', mse_avg)
        plotter.plot('PSNR avg', psnr_avg)
        if(final):
            print('final iteration %d SSIM avg: %.2f MSE avg: %.2f' % (iteration, ssim_avg, mse_avg)) 
            ssim_vals_list = ssim_vals.eval()  
            mse_vals_list = mse_vals.eval()    
            psnr_vals_list = psnr_vals.eval() 
            print(ssim_vals_list) # save it in nohup.out
            print(mse_vals_list)
            print(psnr_vals_list)
    else:
         imsaver.save_images(samples_255.reshape((BATCH_SIZE, IM_DIM, IM_DIM)), 'samples_{}.jpg'.format(frame)) # , alternate_viz=True)
    if(MODE == 'vae'):
         noise_samples = session.run(more_noise_samples) # [0,1]
         noise_samples_255 = ((noise_samples)*(255.)).astype('uint8') # [0,255] 
         imsaver.save_images(noise_samples_255.reshape((BATCH_SIZE, IM_DIM, IM_DIM)), 'noise_samples_{}.jpg'.format(frame)) # , alternate_viz=True)
    if(final):  # calculate accuracy of samples (mnist classificator)
        if(MODE == 'cond_ordered'):
            _fixed_labels = np.zeros((fixed_labels_array.size, N_LABELS))
            _fixed_labels[np.arange(fixed_labels_array.size), fixed_labels_array] = 1 # np to one-hot
        else:
            _fixed_labels = np.zeros((sorted_labels.size, N_LABELS))
            _fixed_labels[np.arange(sorted_labels.size), sorted_labels] = 1 # np to one-hot
        accu = session.run(accuracy, feed_dict={x: samples, y: _fixed_labels}) 
        print('Accuracy at step %d: %s' % (iteration, accu))     

# -------------------------------- Train loop ---------------------------------------------

config = tf.ConfigProto()  # dynamic memory growth
config.gpu_options.allow_growth = True

with tf.Session(config=config) as session:

    # Init variables
    if(START_ITER > 0):
         saver.restore(session, restore_path) # Restore variables from saved session.
         print("Model restored.")
         plotter.restore(START_ITER)  # makes plots start from 0
         session.run(fixed_noise.initializer)
    else:
        session.run(init_op)	
        session.run(fixed_noise.initializer)
    print(fixed_noise.eval())

    # Classificator Training
    for i in range(1000):
        if i % 100 == 0:  # calc accuracy every 100 steps during training
            _data, _labels = next(test_generator)
            labels = np.zeros((_labels.size, N_LABELS))
            labels[np.arange(_labels.size),_labels] = 1 # np: to one-hot
            feed = {x: _data, y: labels}
            acc, loss = session.run([accuracy, cross_entropy], feed_dict=feed)
            print('Accuracy at step %s: %s - loss: %f' % (i, acc, loss))
            if(acc >= 0.98): # wont get much better than 98%
                break 
        else:		# train classificator
            _data, _labels = next(gen)
            labels = np.zeros((_labels.size, N_LABELS))
            labels[np.arange(_labels.size),_labels] = 1 # np to one-hot
            feed = {x: _data, y: labels}
        session.run(train_step, feed_dict=feed)

    overall_start_time = time.time()  
    
    summary_writer = tf.summary.FileWriter(log_dir, graph=session.graph)

    # Network Training
    for iteration in range(START_ITER, ITERS):  # START_ITER: 0 or from last checkpoint
        start_time = time.time()
        # Train generator (and Encoder)
        if (iteration > 0):
            _data, _labels = next(gen)
            if(MODE == 'enc'):
                _, summary_str = session.run([gen_train_op, merged_summary_op_g], feed_dict={real_data: _data, condition_data: _labels})
            elif(MODE == 'cond' or MODE == 'cond_ordered'):
                _, summary_str = session.run([gen_train_op, merged_summary_op_g], feed_dict={condition_data: _labels})
            elif(MODE == 'vae'):
                _, summary_str = session.run([train_op, merged_summary_op_vae], feed_dict={real_data: _data})
            else:
                _, summary_str = session.run([gen_train_op, merged_summary_op_g])
            summary_writer.add_summary(summary_str, iteration)
        if(MODE != 'vae'):
            # Train discriminator
            for i in range(DISC_ITERS):
                _data, _labels = next(gen) 
                if(MODE == 'plain'): 
                    _disc_cost, _, summary_str = session.run([disc_cost, disc_train_op, merged_summary_op_d], feed_dict={real_data: _data})
                else:
                    _disc_cost, _, summary_str = session.run([disc_cost, disc_train_op, merged_summary_op_d], feed_dict={real_data: _data, condition_data: _labels})
            summary_writer.add_summary(summary_str, iteration)
            plotter.plot('train disc cost', _disc_cost)
        iteration_time = time.time() - start_time
        plotter.plot('time', iteration_time)

        # Validation: Calculate validation loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            dev_vae_losses = []
            _data, _labels = next(dev_generator)  
            if(MODE == 'plain'): 
                _dev_disc_cost, _dev_gen_cost = session.run([disc_cost, gen_cost], feed_dict={real_data: _data})
                print("Step %d: D: loss = %.7f G: loss=%.7f " % (iteration, _dev_disc_cost, _dev_gen_cost))
                dev_disc_costs.append(_dev_disc_cost)
                plotter.plot('dev disc cost', np.mean(dev_disc_costs))
            elif(MODE == 'vae'): 
                _dev_vae_loss = session.run(vae_loss, feed_dict={real_data: _data})
                print("Step %d: total VAE loss = %.7f" % (iteration, _dev_vae_loss))
                dev_vae_losses.append(_dev_vae_loss)
                plotter.plot('vae loss', np.mean(dev_vae_losses))           
            else:
                _dev_disc_cost, _dev_gen_cost = session.run([disc_cost, gen_cost], feed_dict={real_data: _data, condition_data: _labels})
                print("Step %d: D: loss = %.7f G: loss=%.7f " % (iteration, _dev_disc_cost, _dev_gen_cost))
                dev_disc_costs.append(_dev_disc_cost)
                plotter.plot('dev disc cost', np.mean(dev_disc_costs))
            generate_image(iteration, False) 

        # Save logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            plotter.flush()

        plotter.tick()
    generate_image(ITERS, True) # outputs ssim and mse vals for all samples, calculate accuracy
    save_path = saver.save(session, restore_path) # Save the variables to disk.
    print("Model saved in path: %s" % save_path)

summary_writer.close() # flushes the outputwriter to disk
        
overall_end_time = time.time()  # time analysis
overall_time = (overall_end_time - overall_start_time)
print("From ", START_ITER, "to ", ITERS," the GAN took ", overall_time, "sec to run")
avg_time_per_iteration = overall_time/ITERS
print('average time per iteration: ', avg_time_per_iteration, 'sec')
overall_time /= 60.
print("From ", START_ITER, "to ", ITERS," the GAN took ", overall_time, "min to run")
