import os
import sys

current_path = os.path.abspath('.')
root_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(root_path)

from sourcecode.ORCA.orca_dataloader_512x512 import *
from sourcecode.train_utils import *



dataset_name="101_ORCA_512x512__"
loss_function="BCELoss" # BCELoss, L1Loss, MSELoss, HuberLoss, SmoothL1Loss
optimizer_algorithm="Adam"

dataset_dir = "../../datasets/ORCA_512x512"
model_dir = "../../models"
result_file_csv = "../../datasets/ORCA_512x512/training/{}_training_accuracy_loss_{}_{}.csv".format(dataset_name, loss_function, optimizer_algorithm)

augmentation_strategy = "random"  # "no_augmentation", "color_augmentation", "inpainting_augmentation", "standard", "random"
augmentation = [None,
                # "horizontal_flip",
                # "vertical_flip",
                # "rotation",
                # "transpose",
                # "elastic_transformation",
                # "grid_distortion",
                # "optical_distortion",
                # "color_transfer",
                # "inpainting",
                'CLAHE', 'Downscale', 'Equalize', 'HueSaturationValue', 'ISONoise', 'MultiplicativeNoise', 'RandomGravel', 'RingingOvershoot', 'Sharpen', 'Blur', 'Defocus', 'GaussianBlur', 'GlassBlur', 'MedianBlur', 'MotionBlur', 'ZoomBlur']

use_cuda = True
start_epoch = 1
n_epochs = 400
batch_size = 1
patch_size = (512, 512)
color_model = "LAB"

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

# loads our u-net based model to continue previous training
# trained_model_version = "ORCA_512x512__Size-512x512_Epoch-280_Images-100_Batch-1__random_8_operations_all"
# trained_model_path = "{}/{}.pth".format(model_dir, trained_model_version)
# model = load_model(file_path=trained_model_path, img_input_size=patch_size, use_cuda=True)

# starts the training from scratch
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
