import os

from sourcecode.Utils.dataloader_utils import is_valid_file
from sourcecode.Utils.logger_utils import logger


def orca_load_dataset(img_dir, img_input_size, dataset_type):

    images = []
    classes = ["tumor"]
    dataset_root_dir = "{}/{}".format(img_dir, dataset_type)
    logger.info("[{}] {}".format(dataset_type, dataset_root_dir))

    for root, d, _ in sorted(os.walk(dataset_root_dir)):

        for cls in sorted([cls for cls in d if cls in classes]):

            class_root_dir = "{}/{}/patch/{}x{}".format(dataset_root_dir, cls, img_input_size[0], img_input_size[1])

            for _, img_dir, _ in sorted(os.walk(class_root_dir)):

                for img_number in sorted(img_n for img_n in img_dir):

                    for patch_type in ["01-roi", "02-non_roi"]:

                        original_dir = "{}/{}/{}/01-original".format(class_root_dir, img_number, patch_type)
                        mask_dir = "{}/{}/{}/02-mask".format(class_root_dir, img_number, patch_type)
                        for _, _, fnames in sorted(os.walk(original_dir)):
                            for fname in sorted(fnames):

                                path_img = os.path.join(original_dir, fname)
                                path_mask = os.path.join(mask_dir, fname)

                                #if is_valid_file(path_img) and first_train.find(fname) < 0:
                                if is_valid_file(path_img):
                                    item = (path_img, path_mask, fname)
                                    images.append(item)
    #first_train = ""
    return images
