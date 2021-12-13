
from utils import make_data_individual,make_data_combined


basedir = r'D:/DATA-STOR/AAPM Dataset'
subdirs = ['/L067','/L096','/L109','/L143','/L192','/L286','/L291','/L310','/L333']
ydir = '//full_1mm'
xdir = '//quarter_1mm'  

#----------------------------Make 40x40 training pateches----------------------


 
individual_name =  '//train_patches_exclude_air_40.h5'
individual_name_combined = '//Train40.h5' 
window_shape = (40,40)  
step_size = 40  
thresh = 0.05

make_data_individual(basedir,subdirs,xdir,ydir,individual_name,window_shape,step_size,thresh)
make_data_combined(basedir, subdirs, individual_name,individual_name_combined)


#----------------------------Make 512x512 test images--------------------------


basedir = r'D:/DATA-STOR/AAPM Dataset'
subdirs = ['/L506']
ydir = '//full_1mm'
xdir = '//quarter_1mm'  
individual_name =  '//test_512.h5'
individual_name_combined = '//Test512.h5' 
window_shape = (512,512)  
step_size = 512  
thresh = 0.05

make_data_individual(basedir,subdirs,xdir,ydir,individual_name,window_shape,step_size,thresh)
make_data_combined(basedir, subdirs, individual_name,individual_name_combined)