import os
import sys

current_path = os.path.abspath('.')
root_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(root_path)

from sourcecode.Utils.oscc_dataloader import *
from sourcecode.Utils.train_utils import *
from sourcecode.Utils.orca_load_dataset_512x512 import *

### Model ###
# loads u-net based model to continue previous training (file name without extension .pth)
#trained_model_version = "ORCA512"
trained_model_version = None # starts the training from scratch

### Configurations ###
start_epoch = 1
n_epochs    = 400
batch_size  = 1
patch_size  = (512, 512)
color_model = "LAB"
use_cuda    = True

first_fname_id = 1662
dataset        = "ORCA512"

#list_loss      = ['BCELoss', 'L1Loss', 'MSELoss', 'HuberLoss', 'SmoothL1Loss']
#list_optimizer = ['Adam', 'Adadelta', 'Adagrad', 'AdamW', 'Adamax', 'ASGD', 'NAdam', 'RAdam', 'RMSprop', 'Rprop', 'SGD']
#list_strategy  = ['no_augmentation', 'color_augmentation', 'inpainting_augmentation', 'geometric', 'distortion',
#                  'standard', 'random', 'solo']

list_loss      = ['HuberLoss']
list_optimizer = ['Adam']
list_strategy  = ['solo']

all_augmentations = ['ShiftScaleRotate',
                     'AdvancedBlur',
                     'ChannelDropout',
                     'ChannelShuffle',
                     'ChromaticAberration',
                     'ColorJitter',
                     'Emboss',
                     'FancyPCA',
                     'GaussNoise',
                     'ImageCompression',
                     'InvertImg',
                     'PlanckianJitter',
                     'Posterize',
                     'RGBShift',
                     'RandomBrightnessContrast',
                     'RandomFog',
                     'RandomGamma',
                     'RandomRain',
                     'RandomShadow',
                     'RandomSnow',
                     'RandomSunFlare',
                     'RandomToneCurve',
                     'Solarize',
                     'Spatter',
                     'Superpixels',
                     'ToGray',
                     'ToSepia',
                     'UnsharpMask',
                     'Affine',
                     'CoarseDropout',
                     'D4',
                     'GridDropout',
                     'Lambda',
                     'PiecewiseAffine',
                     'RandomGridShuffle',
                     'XYMasking',
                     ##'Normalize',
                     ##'LongestMaxSize',
                     ##'MixUp',
                     ##'PadIfNeeded',
                     ##'RandomCropFromBorders',
                     ##'RandomScale',
                     ##'SmallestMaxSize',
                     ##'FDA',
                     ##'PixelDistributionAdaptation',
                     ##'TemplateTransform',
                    ]


### Directories and files ###
dataset_dir = "../../datasets/ORCA_512x512"
model_dir = "../../models"

### Model saving frequency
#model_saving_frequency = None          # No save
#model_saving_frequency = ('all', 0)    # Save all
#model_saving_frequency = ('every', 10) # Save every 10 epochs
model_saving_frequency = ('last', 2)    # Save just the last 3 epochs


################################################################################################################


for loss_function in list_loss:
    for optimizer_algorithm in list_optimizer:
        for augmentation_strategy in list_strategy:
            if augmentation_strategy in ['no_augmentation', 'color_augmentation', 'inpainting_augmentation']:
                augmentation = [None]
            elif augmentation_strategy in ['standard', 'random']:
                augmentation = [["horizontal_flip", "vertical_flip", "rotation", "transpose", "grid_distortion", "optical_distortion", "color_transfer", "inpainting"]]
            elif augmentation_strategy == 'geometric':
                augmentation = [["horizontal_flip", "vertical_flip", "rotation", "transpose"]]
            elif augmentation_strategy == 'distortion':
                augmentation = [["grid_distortion", "optical_distortion"]]
            elif augmentation_strategy == 'solo':
                augmentation = [[aug] for aug in all_augmentations]
            else:
                print('Invalid augmentation strategy: {}'.format(augmentation_strategy))
                exit(1)

            for i in range(0, len(augmentation)):
                dataset_name = "{}_{}".format(first_fname_id, dataset)
                first_fname_id = first_fname_id + 1
                result_file_csv = dataset_dir + "/training/{}_training_accuracy_loss_{}_{}.csv".format(dataset_name,
                                                                                                       loss_function,
                                                                                                       optimizer_algorithm)
                print("{} {} {} {} {}".format(dataset_name, loss_function, optimizer_algorithm, augmentation_strategy, augmentation[i]))


                # load images
                dataloaders = create_dataloader(samples_function=orca512_load_dataset,
                                                tile_size="{}x{}".format(patch_size[0], patch_size[1]),
                                                batch_size=batch_size,
                                                shuffle=False,
                                                img_input_size=patch_size,
                                                img_output_size=patch_size,
                                                dataset_dir=dataset_dir,
                                                color_model=color_model,
                                                augmentation=augmentation[i],
                                                augmentation_strategy=augmentation_strategy,
                                                start_epoch=start_epoch,
                                                validation_split=0.0,
                                                use_cuda=use_cuda)


                # load a trained model or define model as None
                if trained_model_version is not None:
                    trained_model_path = "{}/{}.pth".format(model_dir, trained_model_version)
                    model = load_model(file_path=trained_model_path, img_input_size=patch_size, use_cuda=True)
                else:
                    model = None


                # train the model
                train_model_with_validation(dataloaders=dataloaders,
                                            model=model,
                                            patch_size=patch_size,
                                            n_epochs=n_epochs,
                                            start_epoch=start_epoch,
                                            batch_size=batch_size,
                                            use_cuda=use_cuda,
                                            output_dir=model_dir,
                                            augmentation_strategy=augmentation_strategy,
                                            augmentation_operations=augmentation[i],
                                            dataset_name=dataset_name,
                                            loss_function=loss_function,
                                            optimizer_algorithm=optimizer_algorithm,
                                            result_file_csv=result_file_csv,
                                            model_saving_frequency=model_saving_frequency)
