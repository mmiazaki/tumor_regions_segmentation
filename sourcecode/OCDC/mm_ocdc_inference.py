import os
import sys

current_path = os.path.abspath('.')
root_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(root_path)

from sourcecode.unet_model import *
from sourcecode.wsi_image_utils import *

import os
import matplotlib.pyplot as plt

import torchvision.transforms.functional as TF
from torchvision import utils
from torch.autograd import Variable

import gc

torch.cuda.empty_cache()
gc.collect()

dataset_dir = "../../datasets/OCDC"
model_dir = "../../models"

color_model = "LAB"
magnification = 0.625
scale = get_scale_by_magnification(magnification)
tile_size = 20
tile_size_original = int(scale * tile_size)
patch_size = (tile_size_original, tile_size_original)

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if use_cuda else "cpu"

# loads our trained fcn model
trained_model_version = "OCDC__Size-640x640_Epoch-400_Images-840_Batch-1__random_9_operations"
trained_model_path = "{}/{}".format(model_dir, '{}.pth'.format(trained_model_version))

if not os.path.isfile(trained_model_path):

    logger.info("Trained model not found: '{}'.".format(trained_model_path))

else:

    model = load_checkpoint(file_path=trained_model_path, img_input_size=patch_size, use_cuda=use_cuda)

    dataset_type = "testing"
    classes = ["tumor"]
    for cls in classes:

        wsi_images_dir = "{}/{}/{}/wsi".format(dataset_dir, dataset_type, cls)
        patch_images_dir = "{}/{}/{}/patch/640x640".format(dataset_dir, dataset_type, cls)
        patch_images_results_dir = "{}/results/{}/{}/{}/patch/640x640".format(dataset_dir, trained_model_version,
                                                                              dataset_type, cls)
        wsi_images_results_dir = "{}/results/{}/{}/{}/wsi".format(dataset_dir, trained_model_version, dataset_type, cls)

        for r, d, f in sorted(os.walk(wsi_images_dir)):
            for wsi_file in sorted(f):

                wsi_image_file = "{}/{}".format(r, wsi_file)
                wsi_image_number = wsi_file.replace(".svs", "")

                file_is_svs = wsi_image_file.lower().endswith('.svs')
                if file_is_svs:

                    logger.info("Processing wsi '{}'".format(wsi_file))
                    if not os.path.isfile(wsi_image_file):
                        logger.info("WSI image not found: '{}'.".format(wsi_file))
                        break

                    # check directory to save image-patches
                    dir_to_save = "{}/{}".format(patch_images_results_dir, wsi_image_number)
                    if not os.path.exists(dir_to_save):
                        os.makedirs("{}/{}".format(wsi_images_results_dir, wsi_image_number))
                        os.makedirs("{}/01-unet_result".format(dir_to_save))

                    # scale down image
                    wsi_image = open_wsi(wsi_image_file)
                    pil_scaled_down_image = scale_down_wsi(wsi_image, magnification, False)
                    np_scaled_down_image = pil_to_np(pil_scaled_down_image)

                    # extract tissue region
                    np_tissue_mask, np_masked_image = extract_normal_region_from_wsi(wsi_image_file,
                                                                                     np_scaled_down_image, None)
                    pil_masked_image = np_to_pil(np_masked_image)

                    # draw the heat grid
                    pil_img_result, heat_grid, number_of_tiles = draw_heat_grid(np_masked_image, tile_size)

                    # save scaled down wsi
                    utils.save_image(TF.to_tensor(pil_scaled_down_image),
                                     '{}/{}/{}.png'.format(wsi_images_results_dir, wsi_image_number, wsi_image_number))
                    utils.save_image(TF.to_tensor(pil_masked_image),
                                     '{}/{}/{}__tissue.png'.format(wsi_images_results_dir, wsi_image_number,
                                                                   wsi_image_number))
                    utils.save_image(TF.to_tensor(pil_img_result),
                                     '{}/{}/{}__tissuegrid.png'.format(wsi_images_results_dir, wsi_image_number,
                                                                       wsi_image_number))

                    # run the model
                    count_tiles = 0
                    count_roi_tiles = 0
                    for idx, (position, row, column, location, size, color) in enumerate(heat_grid):

                        if color != GREEN_COLOR:

                            count_tiles += 1

                            r_s = row * tile_size
                            r_e = r_s + tile_size
                            c_s = column * tile_size
                            c_e = c_s + tile_size
                            np_tile_masked = np_masked_image[r_s:r_e, c_s:c_e]

                            # only tile with valid size
                            if np_tile_masked.shape[0] == tile_size and np_tile_masked.shape[1] == tile_size:

                                # read the tile from the original wsi image
                                pil_input_tile, np_input_tile = read_region(wsi_image_file, column, row, magnification,
                                                                            tile_size)

                                # run the model
                                if color_model == "LAB":
                                    np_input_tile = rgb_to_lab(np_input_tile)

                                X = torch.from_numpy(np_input_tile).permute(2, 0, 1).float()
                                X = Variable(X.unsqueeze(0)).to(device) if use_cuda else X.unsqueeze(0)
                                y_hat = model(X).detach().cpu().squeeze(0)
                                # y_hat = model(X).squeeze(0)
                                output_tile = y_hat[0]
                                np_output_tile = output_tile.squeeze(0).detach().cpu().numpy()
                                #

                                # only tiles that something was found by model
                                if np.any(np.unique(np_output_tile >= 0.1)):
                                    # save the output image-patch results
                                    utils.save_image(output_tile, '{}/01-unet_result/{}_r{}c{}.png'.format(dir_to_save,
                                                                                                           wsi_image_number,
                                                                                                           row, column))

                                    np_output_tile[np_output_tile > 0] = 1
                                    count_roi_tiles += 1

                    logger.info("\t '{}/{}/{}' tiles identified as ROI by model".format(count_roi_tiles, count_tiles,
                                                                                        len(heat_grid)))
                    logger.info("-")

                    # save scaled down wsi
                    utils.save_image(TF.to_tensor(pil_scaled_down_image),
                                     '{}/{}/{}.png'.format(wsi_images_results_dir, wsi_image_number, wsi_image_number))