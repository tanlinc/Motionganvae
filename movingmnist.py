import numpy as np
import os
from skimage.transform import resize

def movingmnist_generator(data, batch_size, im_size, shuffle):
    # data.shape = (20, ..., 64, 64)
    frames_per_seq = data.shape[0]
    num_seqs = data.shape[1]
    data = np.reshape(data, [frames_per_seq, num_seqs, 4096]) # (20, ..., 4096)
 
    if(im_size == 32): # downsampling to 32x32
        data_new = np.zeros([frames_per_seq, num_seqs, 32*32])
        for frame in range(0, frames_per_seq):
            for seq in range (0, num_seqs):
                x = data[frame, seq,:].reshape(64,64)	
                image = resize(x, (32, 32)) # already normalizes to [0,1]
                x = image.flatten()	
                data_new[frame, seq,:] = x
        data = data_new
    elif(im_size != 64):
        print('image size not supported! Choose 64x64 or 32x32!')
    else:
        data = (data/255.).astype('float32') # normalize to [0,1]
    data = data.astype('float32') 
    data = np.transpose(data, [1,0,2]) # transpose for shuffling!!! 
    # (7000, 20, 4096)  # (2000, 20, 4096)  # (1000, 20, 4096)  # print(data.shape) for IM_DIM 64
    # (7000, 20, 1024)  # (2000, 20, 1024)  # (1000, 20, 1024)  # print(data.shape) for IM_DIM 32

    def get_epoch():
        NUM_BATCHES = np.round(num_seqs/batch_size).astype('uint8') # for whole 10000, b=50: 200

        while True:
            if(shuffle):
                np.random.shuffle(data) # shuffles first dim
 
            for seq_ctr in range (0, NUM_BATCHES):
                for frame in range(0,frames_per_seq-5): # always 6 frames in a row from the 20 frames, so 0-5, ... 14-19
                    batch = data[seq_ctr*batch_size:(seq_ctr+1)*batch_size, frame:(frame+6), :]
                    yield np.copy(batch)

    return get_epoch            

def load(batch_size = 50, test_batch_size = 50, imsize = 64): 
    filepath = '/home/linkermann/Desktop/MA/data/movingMNIST/mnist_test_seq.npy'

    if not os.path.isfile(filepath):
        print("Couldn't find movingMNIST dataset")

    dat = np.load(filepath) 
    # print(dat.shape) = (20, 10000, 64, 64)
    # TODO: generate/load other files as dev and test data!

    # split dataset into training, test and validation sets
    train_data, dev_data, test_data = dat[:,0:7000,:,:], dat[:,7000:9000,:,:], dat[:,9000:10000,:,:]

    return (
        movingmnist_generator(train_data, batch_size, shuffle = True, im_size = imsize), 
        movingmnist_generator(dev_data, test_batch_size, shuffle = True, im_size = imsize), 
        movingmnist_generator(test_data, test_batch_size, shuffle = False, im_size = imsize)
    )

#if __name__ == '__main__':
#    train_gen, dev_gen, test_gen = load(50, 50)	# load sets into global vars
#    gen = train_gen()		# init iterator for training set
#    
#    data = next(gen)
#    print(data.shape) # (3, 50, 4096)
