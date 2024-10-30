import os
import sys


current_path = os.path.abspath('.')
root_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(root_path)

from sourcecode.ORCA.orca_dataloader_512x512 import *
from sourcecode.train_utils import *

dataset_dir = "../../datasets/ORCA"
model_dir = "../../models"
result_file_csv = "../../datasets/ORCA/training/orca_training_accuracy_loss_all.csv"

augmentation_strategy = "random" # "no_augmentation", "color_augmentation", "inpainting_augmentation", "standard", "random"
augmentation = [None,
				"horizontal_flip", 
				"vertical_flip", 
				"rotation", 
				"transpose", 
				"elastic_transformation", 
				"grid_distortion", 
				"optical_distortion", 
				"color_transfer", 
				"inpainting"]
#[None, "horizontal_flip", "vertical_flip", "rotation", "transpose", "elastic_transformation", "grid_distortion", "optical_distortion", "color_transfer", "inpainting"]

use_cuda = True
start_epoch = 1
n_epochs = 100
batch_size = 1
patch_size = (640, 640)
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
#    trained_model_version = "ORCA__Size-640x640_Epoch-22_Images-4181_Batch-1__random_9_operations_distortion"
#    trained_model_path = "{}/{}.pth".format(model_dir, trained_model_version)
#    model = load_checkpoint(file_path=trained_model_path, img_input_size=patch_size, use_cuda=True)

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
							result_file_csv=result_file_csv)
