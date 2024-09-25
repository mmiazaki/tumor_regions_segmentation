import os
import sys

current_path = os.path.abspath('.')
root_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(root_path)

from sourcecode.ORCA.orca_train import *
from sourcecode.ORCA.orca_dataloader_512x512 import *

import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision import utils
from datetime import datetime
from skimage import measure

dataset_dir = "../../datasets/ORCA_512x512"
model_dir = "../../models"

use_cuda=True
batch_size = 1
patch_size = (512, 512)
color_model = "LAB"
threshold_prob = 0.50
threshold_itc = 200/(0.243 * pow(2, 5))

# loads our trained fcn model
load_models = [
    # "ORCA_512x512__Size-512x512_Epoch-400_Images-100_Batch-1__random_9_operations_all",
    # '022-ORCA512-BCELoss-random9',
    # '023-ORCA512-BCELoss-random8',
    # '024-ORCA512-BCELoss-random9',
    # '025-ORCA512-BCELoss-random8',
    # '026-ORCA512-L1Loss-random9',
    # '027-ORCA512-L1Loss-random8',
    # '028-ORCA512-MSELoss-random8',
    # '029-ORCA512-MSELoss-random9',
    # '030-ORCA512-HuberLoss-random9',
    # '031-ORCA512-SmoothL1Loss-random9',
    # '032-ORCA512-SmoothL1Loss-random8',
    # '033-ORCA512-HuberLoss-random8',
    # '034-ORCA512-BCELoss-random9',
    # '035-ORCA512-BCELoss-random8',
    # '036-ORCA512-BCELoss-random9',
    '100_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_25_operations',
    '101_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_16_operations',
    '102_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_18_operations',
    '103_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_24_operations',
    '104_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_standard_8_operations',
    '105_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_inpainting_augmentation']





# Checking for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if use_cuda else "cpu"
logger.info('Runing on: {}'.format(device))

for trained_model_version in load_models:
    print(':::::: {} ::::::'.format(trained_model_version))
    dataloaders = create_dataloader(batch_size=batch_size,
                                    shuffle=False,
                                    dataset_dir=dataset_dir,
                                    color_model=color_model)

    dataset_train_size = len(dataloaders['train'].dataset)
    dataset_test_size = len(dataloaders['test'].dataset)

    trained_model_path="{}/{}".format(model_dir, '{}.pth'.format(trained_model_version))
    model = load_checkpoint(file_path=trained_model_path, img_input_size=patch_size, use_cuda=use_cuda)
    cont = 0
    for batch_idx, (images, masks, fname, original_size) in enumerate(dataloaders['test']):

        X = Variable(images).to(device) if use_cuda else images
        logger.info('Batch {}: {}/{} images: {} masks: {} {}'.format(
                    (batch_idx+1),
                    (batch_idx+1) * len(images),
                    len(dataloaders['test'].dataset),
                    images.shape,
                    masks.shape,
                    datetime.now().strftime('%d/%m/%Y %H:%M:%S')))

        #X_numpy = X.cpu().numpy()
        y_hat = model(X).detach().cpu().squeeze(0)
        cls = "tumor"

        input_image = transforms.ToPILImage()(X.squeeze(0).cpu())
        output_mask = basic_threshold(y_hat[0].detach().cpu().numpy(), threshold=threshold_prob, output_type="uint8")

        predicted_labels = measure.label(output_mask, connectivity=2)
        output_mask = np.zeros((output_mask.shape[0], output_mask.shape[1]))
        labels = np.unique(predicted_labels)
        properties = measure.regionprops(predicted_labels)
        for lbl in range(1, (np.max(labels)+1)):
            major_axis_length = properties[lbl-1].major_axis_length
            if major_axis_length > threshold_itc:
                output_mask[predicted_labels == lbl] = 1

        input_image_rgb = lab_to_rgb(pil_to_np(input_image))
        roi_image = blend_image(np_to_pil(input_image_rgb), np_to_pil(output_mask), foreground='red', alpha=0.6, inverse=True)

        # results dir
        wsi_image_number = fname[0].split("_")[0] + "_" + fname[0].split("_")[1]
        patch_images_results_dir = "{}/results/{}/testing/{}/patch/{}x{}/{}".format(dataset_dir, trained_model_version, cls, patch_size[0], patch_size[1], wsi_image_number)
        results_output_dir = "{}/01-unet_result".format(patch_images_results_dir)
        if not os.path.exists(results_output_dir):
            os.makedirs(results_output_dir)

        results_roi_dir = "{}/02-roi".format(patch_images_results_dir)
        if not os.path.exists(results_roi_dir):
            os.makedirs(results_roi_dir)

        # save the results
        patch_img_name = fname[0]
        utils.save_image(y_hat[0], '{}/{}'.format(results_output_dir, patch_img_name))
        utils.save_image(TF.to_tensor(roi_image), '{}/{}'.format(results_roi_dir, patch_img_name))
