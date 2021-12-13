import numpy as np
import glob
import SimpleITK as itk
import math
from skimage.metrics import structural_similarity as ssim
import h5py
from skimage.util import view_as_windows        

def load_images(path):
    

    dcm_path = glob.glob(path +  '/*.IMA' , recursive=True)
    dcm_path.sort()
    n_images = len(dcm_path)
    
    
    imgs=np.zeros((n_images,512,512))
    for i in range(n_images):
        tem_img = itk.ReadImage(dcm_path[i])
        img = itk.GetArrayFromImage(tem_img)
        # img = preprocess_image(img)   
        imgs[i] = (img + 1024.0) / 4095.0
    return imgs


def preprocess_image(img):
    out = np.zeros(img.shape)

    for n,val in enumerate(img):
        out[n] = 2*((val-val.min())/(val.max()-val.min()))-1

    out = np.nan_to_num(out)
    return out.astype(np.float32)

# def deprocess_image(gen_img, label):
    
#     for n,val in enumerate(gen_img):
#         img = 0.5*(gen_img + 1) * (label.max()-label.min()) + label.min()
#         print(label.min(),label.max())
#     return img.astype('float32')

def extract_patches(im, window_shape=(40,40),stride=40):
    num_imgs = im.shape[0]
    p = [None] * num_imgs
    for i in range(num_imgs):
        p[i] = np.vstack(view_as_windows(im[i,:,:], window_shape, stride))
        
    return np.vstack(p)
 
def PSNR_SSIM(labels_test,labels_pred):       
    # calculate PSNR
    diff = labels_test-labels_pred
    diff = diff.flatten('C')
    rmse = math.sqrt( np.mean(diff ** 2.) )
    psnr = 20*math.log10(1.0/rmse)
    
    # Calculate SSIM
    ssim_val = 0
    # print(type(labels_pred))
    # print(type(labels_test))

    for i in range (labels_test.shape[0]):
        ssim_val = ssim (labels_test[i,:,:,0], labels_pred[i,:,:,0])
        ssim_val = ssim_val/labels_test.shape[0]
    return psnr , ssim_val    

    
def exclude_air(x_patches,y_patches,thresh=-0.9):
    num_patches = x_patches.shape[0]
    indicies = []
    for i in range (num_patches):
        if ((np.mean(x_patches[i,:,:])) < thresh):
            indicies.append(i)
                      
    x_r = np.delete(x_patches,indicies,0)
    y_r = np.delete(y_patches,indicies,0)       
    
    return x_r,y_r
            

def read_hdf5(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        labels = np.array(hf.get('label'))
   
        return data,labels

def write_hdf5(data,labels, output_filename):
    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=data, shape=data.shape)
        h.create_dataset('label', data=labels, shape=labels.shape) 
        
        
def make_data_individual(basedir,subdirs,xdir,ydir,individual_name,window_shape,step_size,thresh=0.9):
    
    for folder in subdirs:
        
        y = load_images(basedir+folder+ydir)
        x = load_images(basedir+folder+xdir)
        y = extract_patches(y,window_shape,step_size)
        x = extract_patches(x,window_shape,step_size)
        y,x = exclude_air(y, x, thresh)
        # y = np.expand_dims(y, axis=3)
        # x = np.expand_dims(x, axis=3)
        savedir = basedir+folder + individual_name
        
        write_hdf5(x,y,savedir)
        
    return

def make_data_combined(basedir,subdirs,individual_name,individual_name_comined):
#    basedir = r'C:\Users\s2ataei\Documents\Dataset'
#    subdirs = ['\L067','\L096','\L143','\L192','\L286','\L291','\L310','\L333']
#    individual_name = '\\train_patches_exclude_air_64'
    Low = [None] * len(subdirs)
    High = [None] * len(subdirs)
    for i,folder in enumerate(subdirs):
        Low[i],High[i] = read_hdf5(basedir+folder+individual_name)
    savedir = './' + individual_name_comined
    
    write_hdf5(np.vstack(Low).astype('float32'),np.vstack(High).astype('float32'),savedir)
    return 




        
# basedir = r'D:/DATA-STOR/AAPM Dataset'
# subdirs = ['/L067','/L096','/L109','/L143','/L192','/L286','/L291','/L310']#,'/L333','/L506']
# ydir = '//full_1mm'
# xdir = '//quarter_1mm'  
# individual_name =  '//train_patches_exclude_air_64.h5'
# individual_name_comined = '//Train64.h5' 
# window_shape=(64,64)  
# step_size=64  
# thresh=-0.9
# make_train_individual(basedir,subdirs,xdir,ydir,individual_name,window_shape,step_size,thresh)
# make_train_combined(basedir, subdirs, individual_name,individual_name_comined)



# thresh = -.9
# y_train = load_images('C:/Users/fnikneja/Desktop/denoise-gan/images/Full/train/')
# x_train = load_images('C:/Users/fnikneja/Desktop/denoise-gan/images/Low/train/')
# y_train_patch = extract_patches(y_train)
# x_train_patch = extract_patches(x_train)
# y_train_patch,x_train_patch = exclude_air(y_train_patch,x_train_patch,thresh)
# y_train_patch = np.expand_dims(y_train_patch, axis=3)
# x_train_patch = np.expand_dims(x_train_patch, axis=3)
# write_hdf5(x_train_patch,y_train_patch,'Train.h5')

    
    

# # plt.Figure()
# # plt.imshow(y_train[1,:,:], cmap='gray')
# # plt.show()

             
       
# y_test = load_images('C:/Users/fnikneja/Desktop/denoise-gan/images/Full/test/')
# x_test = load_images('C:/Users/fnikneja/Desktop/denoise-gan/images/Low/test/')
# y_test = np.expand_dims(y_train, axis=3)
# x_test = np.expand_dims(x_train, axis=3)
# write_hdf5(x_test,y_test,'Test.h5')      
    