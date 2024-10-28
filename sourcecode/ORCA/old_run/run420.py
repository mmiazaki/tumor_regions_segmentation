import os
import sys

current_path = os.path.abspath('.')
root_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(root_path)

from sourcecode.ORCA.orca_dataloader_512x512 import *
from sourcecode.train_utils import *


### Model ###
# loads our u-net based model to continue previous training
#trained_model_version = "" # file name without extension .pth
trained_model_version = None # starts the training from scratch

### Configurations ###
use_cuda = True
start_epoch = 1
n_epochs = 400
batch_size = 1
patch_size = (512, 512)
color_model = "LAB"

dataset_name="420_ORCA512" # prefix name used in the model file
loss_function="SmoothL1Loss" # BCELoss, L1Loss, MSELoss, HuberLoss, SmoothL1Loss
optimizer_algorithm="Adam"

# "no_augmentation"        : without any augmentation
# "color_augmentation"     : color transfer augmentation
# "inpainting_augmentation": inpainting augmentation
# "standard"               : it uses all augmentations, sequentially one by one in each epoch
# "random"                 : (RCAug) it randomly chooses if each augmentation will be used (50% chance for each augmentation)
# "solo"                   : it just uses the first available augmentation in the list (not None)
augmentation_strategy = "solo"

augmentation = [None,
                #"horizontal_flip",
                #"vertical_flip",
                #"rotation",
                #"transpose",
                #"elastic_transformation",
                #"grid_distortion",
                #"optical_distortion",
                #"color_transfer",
                #"inpainting",
                #'CLAHE',
                #'Downscale',
                #'Equalize',
                #'HueSaturationValue',
                #'ISONoise',
                #'MultiplicativeNoise',
                #'RandomGravel',
                #'RingingOvershoot',
                #'Sharpen',
                #'Blur',
                #'Defocus',
                #'GaussianBlur',
                #'GlassBlur',
                #'MedianBlur',
                #'MotionBlur',
                #'ZoomBlur',
                #'Morphological',
                #'PixelDropout',
                #'Rotate',
                #'SafeRotate',
                #'Perspective',
                #'ShiftScaleRotate',
                #'AdvancedBlur',
                #'ChannelDropout',
                #'ChannelShuffle',
                #'ChromaticAberration',
                #'ColorJitter',
                #'Emboss',
                #'FancyPCA',
                #'GaussNoise',
                #'ImageCompression',
                #'InvertImg',
                ###'Normalize',
                #'PlanckianJitter',
                #'Posterize',
                #'RGBShift',
                #'RandomBrightnessContrast',
                #'RandomFog',
                #'RandomGamma',
                #'RandomRain',
                #'RandomShadow',
                'RandomSnow',
                #'RandomSunFlare',
                #'RandomToneCurve',
                #'Solarize',
                #'Spatter',
                #'Superpixels',
                #'ToGray',
                #'ToSepia',
                #'UnsharpMask',
                #'Affine',
                #'CoarseDropout',
                #'D4',
                #'GridDropout',
                #'Lambda',
                ###'LongestMaxSize',
                ###'MixUp',
                ###'PadIfNeeded',
                #'PiecewiseAffine',
                ###'RandomCropFromBorders',
                #'RandomGridShuffle',
                ###'RandomScale',
                ###'SmallestMaxSize',
                #'XYMasking',
                ###'FDA',
                ###'PixelDistributionAdaptation',
                ###'TemplateTransform',
                ]


### Directories and files ###
dataset_dir = "../../datasets/ORCA_512x512"
model_dir = "../../models"
result_file_csv = "../../datasets/ORCA_512x512/training/{}_training_accuracy_loss_{}_{}.csv".format(dataset_name, loss_function, optimizer_algorithm)




################################################################################################################

# load images
dataloaders = create_dataloader(tile_size="{}x{}".format(patch_size[0], patch_size[1]),
                                batch_size=batch_size,
                                shuffle=False,
                                img_input_size=patch_size,
                                img_output_size=patch_size,
                                dataset_dir=dataset_dir,
                                color_model=color_model,
                                augmentation=augmentation,
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
                            n_epochs=n_epochs,
                            start_epoch=start_epoch,
                            use_cuda=use_cuda,
                            augmentation_strategy=augmentation_strategy,
                            output_dir=model_dir,
                            augmentation_operations=augmentation,
                            dataset_name=dataset_name,
                            loss_function=loss_function,
                            optimizer_algorithm=optimizer_algorithm,
                            result_file_csv=result_file_csv)
