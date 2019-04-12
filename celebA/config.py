config_celebA = {};

config_celebA['dataset_dir'] = '/home/akanksha/Desktop/prateek/git/Adversarial-VAE-/celebA-experiments/img_align_celeba/';
config_celebA['dataset_name'] = 'CelebA';
config_celebA['enc_lr'] = 0.0005;
config_celebA['gen_lr'] = 0.0005;
config_celebA['disc_lr'] = 0.0005;
config_celebA['code_disc_lr'] = 0.0005;
config_celebA['batch_size'] = 64;
config_celebA['n_epoch'] = 100;
config_celebA['z_dim'] = 128;
config_celebA['lamda_enc'] = 1;
config_celebA['lamda_gen'] = 5;
config_celebA['img_height'] = 64;
config_celebA['img_width'] = 64;
config_celebA['num_channels'] = 3;
config_celebA['train_min_filenum'] = 1;
config_celebA['train_max_filenum'] = 162770;
config_celebA['val_min_filenum'] = 162771;
config_celebA['val_max_filenum'] = 182637;
config_celebA['original_crop_dir'] = './original-crop/';
config_celebA['generated_crop_dir'] = './generated-crop';
config_celebA['celebA_crop'] = 'closecrop'#'closecrop'
config_celebA['crop_style'] = 'closecrop'#resizecrop for AGE




