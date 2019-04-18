"""
@author: linkermann
"""
import os, sys
sys.path.append(os.getcwd())
import tflib.save_images as imsaver
import tflib.plot as plotter
import tflib.movingmnist as movingmnistloader
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as lays


# --- parameter ------------------------------------------------------------------------------
EXPERIMENT = "cgan_32"  # path to results folder to save/restore
MODE = 'cond' # choose between 'plain', 'cond', 'enc', 'vae'
#'cond' includes conditional input (labels) to G and D [conditional GAN], output input ims compared to generated
# 'enc' includes Encoder before G (with mean&var sampling) as well as conditional input(labels) to D
# 'vae' comprises Encoder and Generator (with mean&var sampling), but no Discriminator

IM_DIM = 32 # number of pixels along x and y (square assumed) # movingmnist: 64, mnist 28
output_dim = IM_DIM*IM_DIM # Number of pixels (movingmnist: 64*64*1) 

BATCH_SIZE = 50 # Batch size needs to divide 50000
LEARNING_RATE = 0.0002 # Adam learning rate for GAN (both G and D)
DIM = 64 # model dimensionality for both Generator and Discriminator, and Encoder
NOISE_DIM = 60 # noise input dim for Generator (plain and cond)
Z_DIM = 60 # for enc/vae: output dim of Encoder, input dim for Generator
LAMBDA = 10 # wgan-gp: Gradient penalty lambda hyperparameter
DISC_ITERS = 5 # wgan-gp: How many discriminator iterations per generator iteration
ITERS = 10000 # How many generator iterations to train for
START_ITER = 0  # Default 0 (start new learning), set accordingly if restoring from checkpoint (100, 200, ...)

restore_path = "/home/linkermann/Desktop/MA/opticalFlow/opticalFlowGAN/results/" + EXPERIMENT + "/model.ckpt" # desktop path
log_dir = "/tmp/tensorflow_movingmnist" # the path of tensorflow's log

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

def Encoder(inputs):   # 32x32 ! todo: batchnorm after inputs and after first two conv layers?
    inputs = tf.reshape(inputs, [BATCH_SIZE, 1, IM_DIM, IM_DIM]) 
    out = lays.conv2d(inputs, DIM, kernel_size = 5, stride = 2, # DIM = 64
        data_format='NCHW', activation_fn=tf.nn.leaky_relu, scope='Enc.1',
        weights_initializer=tf.initializers.he_uniform(), reuse=tf.AUTO_REUSE)
    out = lays.conv2d(out, 2*DIM, kernel_size = 5, stride = 2, # 2*DIM = 128
        data_format='NCHW', activation_fn=tf.nn.leaky_relu, scope='Enc.2',
        weights_initializer=tf.initializers.he_uniform(), reuse=tf.AUTO_REUSE)
    out = lays.conv2d(out, 4*DIM, kernel_size = 5, stride = 2, # 4*DIM = 256
        data_format='NCHW', activation_fn=tf.nn.leaky_relu, scope='Enc.3',
        weights_initializer=tf.initializers.he_uniform(), reuse=tf.AUTO_REUSE)
    #out = lays.conv2d(out, 8*DIM, kernel_size = 5, stride = 2, # new layer 8*DIM
    #    data_format='NCHW', activation_fn=tf.nn.leaky_relu, scope='Enc.4',
    #    weights_initializer=tf.initializers.he_uniform(), reuse=tf.AUTO_REUSE)
    out = tf.reshape(out, [-1, 4*4*4*DIM]) # adjust
    mean = lays.fully_connected(out, Z_DIM, activation_fn=None, scope='Enc.Mean',
        weights_initializer=tf.initializers.glorot_uniform(), reuse=tf.AUTO_REUSE)
    variance = lays.fully_connected(out, Z_DIM, activation_fn=None, scope='Enc.Var',
        weights_initializer=tf.initializers.glorot_uniform(), reuse=tf.AUTO_REUSE)
    return mean, variance  # shape of both: [BATCH_SIZE, Z_DIM]) 

def sample_z(shape, mean, variance): 
    variance = tf.exp(0.5 * variance) # variance has to be positive
    noise = tf.random_normal(shape, mean=0., stddev=1.) # sample from N(0,1)
    return (noise * variance) + mean  # change scale & location  

def Generator_32(n_samples, conditions=None, mean=None, variance=None, noise=None):  
    # (MODE == 'enc' or MODE == 'vae'):          # input: mean and var
    if noise is None: # sample from Gaussian 
        noise = tf.random_normal([n_samples, NOISE_DIM], mean= 0.0, stddev = 1.0) 
    if (MODE == 'cond'):       # input: last frame 
        inputs = tf.concat([noise, conditions], 1) # to: (BATCH_SIZE, 60 + 32*32) 
    elif (MODE == 'plain'):
        inputs = noise
    else:
        if(mean != None and variance != None):
            inputs = sample_z([n_samples, Z_DIM], mean, variance)
    out = lays.fully_connected(inputs, 4*4*4*DIM, reuse = tf.AUTO_REUSE, # expansion
        weights_initializer=tf.initializers.glorot_uniform(), scope = 'Gen.Input')
    out = tf.reshape(out, [-1, 4*DIM, 4, 4])  
    out = tf.transpose(out, [0,2,3,1], name='NCHW_to_NHWC')
    out = lays.conv2d_transpose(out, 2*DIM, kernel_size=5, stride=2, scope='Gen.1',
        weights_initializer=tf.initializers.he_uniform(), reuse=tf.AUTO_REUSE, activation_fn=tf.nn.leaky_relu)
    out = lays.conv2d_transpose(out, DIM, kernel_size=5, stride=2, scope='Gen.2',
        weights_initializer=tf.initializers.he_uniform(), reuse=tf.AUTO_REUSE, activation_fn=tf.nn.leaky_relu)
    out = lays.conv2d_transpose(out, 1, kernel_size=5, stride=2, scope='Gen.4',
        weights_initializer=tf.initializers.he_uniform(), reuse=tf.AUTO_REUSE,
        activation_fn=tf.nn.sigmoid) # sigmoid to get values between (0,1)
    out = tf.transpose(out, [0,3,1,2], name='NHWC_to_NCHW')
    return tf.reshape(out, [BATCH_SIZE, output_dim])

def Discriminator_32(inputs, conditions=None):
    if(MODE == 'plain'):               # inputs: [BATCH_SIZE, output_dim]
        ins = tf.reshape(inputs, [BATCH_SIZE, 1, IM_DIM, IM_DIM]) 
    elif(MODE == 'cond' or MODE == 'enc'):              # input: last frame [BATCH_SIZE, output_dim] 
        ins = tf.reshape(inputs, [BATCH_SIZE, 1, IM_DIM, IM_DIM]) 
        conds = tf.reshape(conditions, [BATCH_SIZE, 1, IM_DIM, IM_DIM]) 
        ins = tf.concat([ins, conds], 1) # to: (BATCH_SIZE, 2, IM_DIM, IM_DIM) 
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
    condition_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, output_dim]) # last frame for D
    mean_data, variance_data = Encoder(condition_data)
    fake_data = Generator_32(BATCH_SIZE, mean=mean_data, variance=variance_data)
    disc_real = Discriminator_32(real_data, conditions=condition_data)
    disc_fake = Discriminator_32(fake_data, conditions=condition_data)
elif(MODE == 'cond'):
    condition_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, output_dim]) # last frame for G and D
    fake_data = Generator_32(BATCH_SIZE, conditions=condition_data)
    disc_real = Discriminator_32(real_data, conditions=condition_data)
    disc_fake = Discriminator_32(fake_data, conditions=condition_data)
elif(MODE == 'vae'):
    mean_data, variance_data = Encoder(real_data)
    fake_data = Generator_32(BATCH_SIZE, mean=mean_data, variance=variance_data)
else:  # plain
    fake_data = Generator_32(BATCH_SIZE)
    disc_real = Discriminator_32(real_data)
    disc_fake = Discriminator_32(fake_data)

fake_image = tf.reshape(fake_data, [BATCH_SIZE, IM_DIM, IM_DIM, 1])
G_image = tf.summary.image("G_out", fake_image)

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
        gradients = tf.gradients(Discriminator_32(interpolates), [interpolates])[0] 
    else:
        gradients = tf.gradients(Discriminator_32(interpolates, conditions=condition_data), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    loss_sum_gp = tf.summary.scalar("D_loss_gp", disc_cost)

    merged_summary_op_d = tf.summary.merge([loss_sum, loss_sum_gp]) # D_prob_sum, D_pro_sum_interpolated])
    merged_summary_op_g = tf.summary.merge([G_loss_sum, G_image]) # G_prob_sum
else:
    # Standard VAE loss
    ## reconstruction loss: pixel-wise L2 loss = mean(square(gen_image - real_im)) # E[log P(X|z)]
    img_loss = tf.reduce_sum(tf.squared_difference(fake_data, real_data), 1)   # axis=[1,2,3]  # has to be sum, not mean!
    mean_img_loss = tf.reduce_mean(img_loss)
    ## latent loss = KL(latent code, unit gaussian) # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian  
    latent_loss = -0.5 * tf.reduce_mean(1. + variance_data - tf.square(mean_data) - tf.exp(variance_data), 1) # log_sigma inst of var_data, negative # better mean than sum
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
train_gen, dev_gen, test_gen = movingmnistloader.load(BATCH_SIZE, BATCH_SIZE, imsize=IM_DIM)	# load sets into global vars

gen = train_gen()			# init iterator for training set
dev_generator = dev_gen()		# init iterator for validation set 
test_generator = test_gen()		# init iterator for test set 

#    data = next(gen)
#    print(data.shape) # (50, 3, 4096)

# --------------------------------- testing: generate samples -----------------------------------------------
fixed_data = next(test_generator) # [0,1] # (50, 3, 4096)
fixed_real_data = (fixed_data[:, 1, :]).reshape((BATCH_SIZE, output_dim)) # current frame # (50, 4096)
fixed_cond_data = (fixed_data[:, 0, :]).reshape((BATCH_SIZE, output_dim)) # only 1 last frame for now # (50, 4096)
fixed_real_data_255 = ((fixed_real_data)*255.).astype('uint8') # [0,255] 
fixed_cond_data_255 = ((fixed_cond_data)*255.).astype('uint8') # [0,255]
fixed_real_data_gt = np.copy(fixed_real_data_255)
fixed_cond_data_gt = np.copy(fixed_cond_data_255)
imsaver.save_images(fixed_real_data_gt.reshape((BATCH_SIZE, IM_DIM, IM_DIM)), 'groundtruth_reals_grey.jpg', alternate_viz=True)
imsaver.save_images(fixed_cond_data_gt.reshape((BATCH_SIZE, IM_DIM, IM_DIM)), 'groundtruth_conds_grey.jpg', alternate_viz=True)

if(START_ITER > 0):    # get noise from saved model: variable, implicit float
    fixed_noise = tf.get_variable("noise", shape=[BATCH_SIZE, NOISE_DIM]) 
else:
    fixed_noise = tf.Variable(tf.random_normal(shape=[BATCH_SIZE, NOISE_DIM]), name='noise') 

if(MODE == 'enc' or MODE == 'vae'):
    fixed_mean, fixed_variance = Encoder(fixed_cond_data)
    fixed_noise_samples = Generator_32(BATCH_SIZE, mean=fixed_mean, variance=fixed_variance)
    _noise_samples = Generator_32(BATCH_SIZE, noise=fixed_noise)
elif(MODE == 'cond'):
    fixed_noise_samples = Generator_32(BATCH_SIZE, conditions=fixed_cond_data, noise=fixed_noise)
elif(MODE == 'plain'):
    fixed_noise_samples = Generator_32(BATCH_SIZE, noise=fixed_noise)

def generate_image(frame, final): # generates a batch of samples next to each other in one image!
    inference_start_time = time.time()  # inference time analysis
    if(MODE == 'cond' or MODE == 'enc'):
        samples = session.run(fixed_noise_samples, feed_dict={condition_data: fixed_cond_data}) # [0,1]
    elif(MODE == 'plain'):
        samples = session.run(fixed_noise_samples) # [0,1]
    elif(MODE == 'vae'):
        samples = session.run(fixed_noise_samples, feed_dict={real_data: fixed_cond_data}) # [0,1]
        samples_noise = session.run(_noise_samples, feed_dict={real_data: fixed_cond_data}) # [0,1]
        noise_samples_255 = ((samples_noise)*255.).astype('uint8') # [0, 1] -> [0,255] 
    inference_end_time = time.time()  # inference time analysis
    inference_time = (inference_end_time - inference_start_time)
    print("The architecture took ", inference_time, "sec for the generation of ", BATCH_SIZE, "images")

    samples_255 = ((samples)*255.).astype('uint8') # [0, 1] -> [0,255] 
    # print('samples 255') # print(samples_255.min()) # print(samples_255.max())

    if(MODE == 'plain'):
        imsaver.save_images(samples_255.reshape((BATCH_SIZE, IM_DIM, IM_DIM)), 'samples_{}.jpg'.format(frame), alternate_viz=True)  
    elif(MODE == 'vae'):
        for i in range(0, BATCH_SIZE):
            samples_255 = np.insert(samples_255, i*2, fixed_cond_data_255[i,:], axis=0) # real (cond digit) next to sample
        imsaver.save_images(samples_255.reshape((2*BATCH_SIZE, IM_DIM, IM_DIM)), 'samples_{}.jpg'.format(frame), alternate_viz=True, conds=True)  
        imsaver.save_images(noise_samples_255.reshape((BATCH_SIZE, IM_DIM, IM_DIM)), 'noise_samples_{}.jpg'.format(frame), alternate_viz=True)  
    else: # if(MODE == 'enc' or MODE == 'cond') add ground truth
        for i in range(0, BATCH_SIZE):
            samples_255 = np.insert(samples_255, i*3, fixed_cond_data_255[i,:], axis=0) # cond left of sample
            samples_255 = np.insert(samples_255, i*3+2, fixed_real_data_255[i,:], axis=0) # real right of sample
        imsaver.save_images(samples_255.reshape((3*BATCH_SIZE, IM_DIM, IM_DIM)), 'samples_{}.jpg'.format(frame), alternate_viz=True, conds=True, gt=True)  

        print("Iteration %d :" % frame)
        # compare generated to real one
        real = tf.reshape(fixed_cond_data, [BATCH_SIZE,IM_DIM,IM_DIM,1])
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

    overall_start_time = time.time()  
    
    summary_writer = tf.summary.FileWriter(log_dir, graph=session.graph)

    # Network Training
    for iteration in range(START_ITER, ITERS):  # START_ITER: 0 or from last checkpoint
        start_time = time.time()
        # Train generator (and Encoder)
        if (iteration > 0):
            _data = next(gen) # [0,1]
            _real_data = (_data[:,1,:]).reshape((BATCH_SIZE, output_dim))  # current frame for now
            _cond_data = (_data[:,0,:]).reshape((BATCH_SIZE, output_dim))  # one last frame for now
            if(MODE == 'enc'):
                _, summary_str = session.run([gen_train_op, merged_summary_op_g], feed_dict={real_data: _real_data, condition_data: _cond_data})
            elif(MODE == 'cond'):
                _, summary_str = session.run([gen_train_op, merged_summary_op_g], feed_dict={condition_data: _cond_data})
            elif(MODE == 'vae'):
                _, summary_str = session.run([train_op, merged_summary_op_vae], feed_dict={real_data: _cond_data})
            else:
                _, summary_str = session.run([gen_train_op, merged_summary_op_g])
            summary_writer.add_summary(summary_str, iteration)
        if(MODE != 'vae'):
            # Train discriminator
            for i in range(DISC_ITERS):
                _data = next(gen) # [0,1]
                _real_data = (_data[:,1,:]).reshape((BATCH_SIZE, output_dim))  # current frame for now
                _cond_data = (_data[:,0,:]).reshape((BATCH_SIZE, output_dim))  # one last frame for now
                if(MODE == 'plain'): 
                    _disc_cost, _, summary_str = session.run([disc_cost, disc_train_op, merged_summary_op_d], feed_dict={real_data: _real_data})
                else:
                    _disc_cost, _, summary_str = session.run([disc_cost, disc_train_op, merged_summary_op_d], feed_dict={real_data: _real_data, condition_data: _cond_data})
            summary_writer.add_summary(summary_str, iteration)
            plotter.plot('train disc cost', _disc_cost)
        iteration_time = time.time() - start_time
        plotter.plot('time', iteration_time)

        # Validation: Calculate validation loss and generate samples every 1000 iters
        if(iteration % 100 == 0): # or iteration < 10):
            dev_disc_costs = []
            dev_vae_losses = []
            _data = next(dev_generator) # [0,1]
            _real_data = (_data[:,1,:]).reshape((BATCH_SIZE, output_dim))  # current frame for now
            _cond_data = (_data[:,0,:]).reshape((BATCH_SIZE, output_dim))  # one last frame for now
            if(MODE == 'plain'): 
                _dev_disc_cost, _dev_gen_cost = session.run([disc_cost, gen_cost], feed_dict={real_data: _real_data})
                print("Step %d: D: loss = %.7f G: loss=%.7f " % (iteration, _dev_disc_cost, _dev_gen_cost))
                dev_disc_costs.append(_dev_disc_cost)
                plotter.plot('dev disc cost', np.mean(dev_disc_costs))
            elif(MODE == 'vae'): 
                _dev_vae_loss = session.run(vae_loss, feed_dict={real_data: _cond_data})
                print("Step %d: total VAE loss = %.7f" % (iteration, _dev_vae_loss))
                dev_vae_losses.append(_dev_vae_loss)
                plotter.plot('vae loss', np.mean(dev_vae_losses))           
            else:
                _dev_disc_cost, _dev_gen_cost = session.run([disc_cost, gen_cost], feed_dict={real_data: _real_data, condition_data: _cond_data})
                print("Step %d: D: loss = %.7f G: loss=%.7f " % (iteration, _dev_disc_cost, _dev_gen_cost))
                dev_disc_costs.append(_dev_disc_cost)
                plotter.plot('dev disc cost', np.mean(dev_disc_costs))
            generate_image(iteration, False) 

        # Save logs every 1000 iters
        if (iteration < 5) or (iteration % 100 == 0):
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
