"""
Image grid saver, based on color_grid_vis from github.com/Newmu

modified by Tanja on 22 Jan 2019
include option for alternate black and grey columns
show sorted test set vertically not horizontally
"""

import numpy as np
import scipy.misc
from scipy.misc import imsave

def save_images(X, save_path, alternate_viz=False, conds=False, gt=False, time=False):

    n_samples = X.shape[0] # 50, 2*50, 3*50
    if(n_samples>50):
        rows = 10  
    else:
        rows = int(np.sqrt(n_samples))
        while n_samples % rows != 0:
            rows -= 1
    nh, nw = rows, (int)(n_samples/rows) # e.g. 10, 10 or 10, 15

    #if X.ndim == 2: # batchsize, flattenedim
    #    X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))
    if X.ndim == 4:
        X = X.transpose(0,2,3,1)  # BCHW -> BHWC
        h, w, c = X[0].shape
        img = np.zeros((h*nh, w*nw, c))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh,w*nw))

    for n, x in enumerate(X):
        ### 2 conditional info and ground truth next frame next to sample (enc)
        if(conds and gt and time and (n%4==0)): # real (dataset)
            i = (int)((n%(nh*4))/4)	# current row
            j = (int)(n/(nh*4)) 	# current col 
            j *= 4
            if(alternate_viz and (j==4 or j==5 or j==6 or j==7 or j==12 or j==13 or j==14 or j==15)): 
                #x[x<100] += 50 # for mnist! 200..
                x[x==0] = 50 	# for moving mnist
            img[i*h:i*h+h, j*w:j*w+w] = x  
        elif(conds and gt and time and (n%4==1)): # fake (sample)
            i = (int)(((n-1)%(nh*4))/4)	# current row
            j = (int)(n/(nh*4)) 	# current col 
            j = (j*4)+1
            if(alternate_viz and (j==4 or j==5 or j==6 or j==7 or j==12 or j==13 or j==14 or j==15)): 
                #x[x<100] += 50 # for mnist! 200..
                x[x==0] = 50 	# for moving mnist
            img[i*h:i*h+h, j*w:j*w+w] = x  
        elif(conds and gt and time and (n%4==2)): # real next frame (dataset)
            i = (int)(((n-2)%(nh*4))/4)	# current row
            j = (int)(n/(nh*4)) 	# current col 
            j = (j*4)+2
            if(alternate_viz and (j==4 or j==5 or j==6 or j==7 or j==12 or j==13 or j==14 or j==15)): 
                x[x==0] = 50 	# for moving mnist
                #x[x<100] += 50 # for mnist! 200..
            img[i*h:i*h+h, j*w:j*w+w] = x
        elif(conds and gt and time and (n%4==3)): # real next frame (dataset)
            i = (int)(((n-3)%(nh*4))/4)	# current row
            j = (int)(n/(nh*4)) 	# current col 
            j = (j*4)+3
            if(alternate_viz and (j==4 or j==5 or j==6 or j==7 or j==12 or j==13 or j==14 or j==15)): 
                x[x==0] = 50 	# for moving mnist
                #x[x<100] += 50 # for mnist! 200..
            img[i*h:i*h+h, j*w:j*w+w] = x  

        ### conditional info and ground truth next frame next to sample (enc, cond)
        elif(conds and gt and (n%3==0)): # real (dataset)
            i = (int)((n%(nh*3))/3)	# current row
            j = (int)(n/(nh*3)) 	# current col 
            j *= 3
            if(alternate_viz and (j==3 or j==4 or j==5 or j==9 or j==10 or j==11)): 
                #x[x<100] += 50 # for mnist! 200..
                x[x==0] = 50 	# for moving mnist
            img[i*h:i*h+h, j*w:j*w+w] = x  
        elif(conds and gt and (n%3==1)): # fake (sample)
            i = (int)(((n-1)%(nh*3))/3)	# current row
            j = (int)(n/(nh*3)) 	# current col 
            j = (j*3)+1
            if(alternate_viz and (j==3 or j==4 or j==5 or j==9 or j==10 or j==11)): 
                #x[x<100] += 50 # for mnist! 200..
                x[x==0] = 50 	# for moving mnist
            img[i*h:i*h+h, j*w:j*w+w] = x  
        elif(conds and gt and (n%3==2)): # real next frame (dataset)
            i = (int)(((n-2)%(nh*3))/3)	# current row
            j = (int)(n/(nh*3)) 	# current col 
            j = (j*3)+2
            if(alternate_viz and (j==3 or j==4 or j==5 or j==9 or j==10 or j==11)): 
                x[x==0] = 50 	# for moving mnist
                #x[x<100] += 50 # for mnist! 200..
            img[i*h:i*h+h, j*w:j*w+w] = x  
        ### conditional info next to sample (vae)
        elif(conds and (n%2==0)): # real (dataset)
            i = (int)((n%(nw*2))/2)	# current row
            j = (int)(n/(nw*2)) 	# current col 
            j *= 2
            if(alternate_viz and (j==2 or j==3 or j==6 or j==7)): 
                #x[x<100] += 50 # for mnist! 200..
                x[x==0] = 50 # for mnist! 200..
            img[i*h:i*h+h, j*w:j*w+w] = x  
        elif(conds and (n%2==1)): # fake (sample)
            i = (int)(((n-1)%(nw*2))/2) # current row
            j = (int)(n/(nw*2)) 	# current col 
            j = (j*2)+1
            if(alternate_viz and (j==2 or j==3 or j==6 or j==7)):
                #x[x<100] += 50 # for mnist! 200..
                x[x==0] = 50 # for mnist!
            img[i*h:i*h+h, j*w:j*w+w] = x  
        ### no cond next to sample (plain or cond_ordered)
        else:	
            i = (int)(n/nw) 	# current row 
            j = n%nw  		# current col
            if(alternate_viz and (j%2==1)): 
                #x[x<100] += 50 	# for mnist!
                x[x==0] = 50 	# for moving mnist
            img[i*h:i*h+h, j*w:j*w+w] = x  # sorted vertically!

    imsave(save_path, img)
