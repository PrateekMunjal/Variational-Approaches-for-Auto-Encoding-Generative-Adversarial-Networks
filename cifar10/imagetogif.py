import imageio
import sys
import os
import re,natsort

images = []
print(sys.argv);
print(sys.argv[1]);
files_dir = sys.argv[1];
temp_files = [];


k=0;
all_files= os.listdir(files_dir);
#sort_files
all_files = natsort.natsorted(all_files)

for file in all_files:
    print(file);

target_name = sys.argv[2];
for filename in all_files:
    images.append(imageio.imread(files_dir + '/'+filename))
imageio.mimsave(target_name+'.gif', images,format='GIF',duration=0.4)