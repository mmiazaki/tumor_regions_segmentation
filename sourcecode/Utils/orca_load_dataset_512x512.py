import os

from sourcecode.Utils.dataloader_utils import is_valid_file
from sourcecode.Utils.logger_utils import logger


def orca512_load_dataset(img_dir, img_input_size, dataset_type):

    images = []
    classes = ["tumor"]
    dataset_root_dir = "{}/{}".format(img_dir, dataset_type)
    logger.info("[{}] {}".format(dataset_type, dataset_root_dir))

    for root, d, _ in sorted(os.walk(dataset_root_dir)):

        for cls in sorted([cls for cls in d if cls in classes]):

            original_dir = "{}/{}/tma".format(dataset_root_dir, cls)
            mask_dir = "{}/lesion_annotations".format(dataset_root_dir)

            for dirpath, dirnames, filenames in sorted(os.walk(original_dir)):
                for fname in sorted(filenames):

                    path_img = os.path.join(original_dir, fname)
                    path_mask = os.path.join(mask_dir, fname.replace(".png", "_mask.png"))

                    #if is_valid_file(path_img) and first_train.find(fname) < 0:
                    if is_valid_file(path_img):
                        item = (path_img, path_mask, fname)
                        images.append(item)
    #first_train = ""
    return images
