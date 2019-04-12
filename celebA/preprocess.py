import pickle
import numpy as np 
import config
from urllib.request import urlretrieve
import os 
import matplotlib.pyplot as plt

opts = config.config_cifar10

dataset_dir = opts['dataset_dir'];

def plot_image(image, isNormalize=True):
    if isNormalize:
        image = denormalize(image);
    plt.imshow(image, origin="upper", cmap='gray');
    plt.show();
    plt.close();

def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_cifar10_batch(dataset_dir, batch_id,isTest=0):
    fname = dataset_dir;
    fname +=  'test_batch' if isTest else 'data_batch_' + str(batch_id)
    print ('Reading from file: ',fname);
    with open(fname, mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')
        
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
        
    return features, labels

def display_stats(dataset_dir, batch_id, sample_id):
    features, labels = load_cifar10_batch(dataset_dir, batch_id)

    print('features.shape: ',features.shape);
    
    print('\nStats of batch #{}:'.format(batch_id))
    print('# of Samples: {}\n'.format(len(features)))
    
    label_names = load_label_names()
    label_counts = dict(zip(*np.unique(labels, return_counts=True)))
    for key, value in label_counts.items():
        print('Label Counts of [{}]({}) : {}'.format(key, label_names[key].upper(), value))
    
    sample_image = features[sample_id]
    sample_label = labels[sample_id]
    
    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))
    
    plt.imshow(sample_image)
    plt.show();

def get_data_from_file(dataset_dir, batch_id,isTest=0):
    features, labels = load_cifar10_batch(dataset_dir, batch_id,isTest)
    return features;

def denormalize(im):
    im += 1; #0 to 2
    im /= 2;
    return im;    

def normalize(im):
    im = im/255. #0-1
    im *= 2;    #0-2
    im -= 1;    #-1 to 1
    return im;

def get_train_and_test_data():
    X_Train = np.zeros((0,32,32,3));
    X_Test = np.zeros((0,32,32,3));
    n_train_batches = 5;

    for batch_id in range(1,n_train_batches+1):
        temp_x_Train = get_data_from_file(dataset_dir,batch_id);
        X_Train = np.vstack((temp_x_Train,X_Train))
    
    X_Test = get_data_from_file(dataset_dir,-1,isTest=1);

    return X_Train, X_Test;


train_file_path = './data/X_train.npy';
test_file_path = './data/X_test.npy';

X_Train, X_Test = get_train_and_test_data();
X_Train = normalize(X_Train);
X_Test = normalize(X_Test);

print(X_Train.shape);
print(X_Test.shape);

#Save X_train and X_test
if not os.path.exists('./data/'):
    os.makedirs('./data/');
np.save(train_file_path,X_Train);
np.save(test_file_path,X_Test);

# load npy files 
X_Train = np.load(train_file_path);
print(X_Train[2].min());
print(X_Train[2].max());
plot_image(X_Train[2]);

