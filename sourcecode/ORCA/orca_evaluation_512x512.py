import os
import sys
import csv

#from scipy import ndimage as nd
from skimage import measure

current_path = os.path.abspath('.')
root_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(root_path)

from sourcecode.ORCA.orca_dataloader_512x512 import *
from sourcecode.wsi_image_utils import *
from sourcecode.evaluation_utils import *

dataset_dir = "../../datasets/ORCA_512x512"
dataset_dir_results = "../../datasets/ORCA_512x512"

batch_size = 1
patch_size = (512, 512)
color_model = "LAB"
dataloaders = create_dataloader(batch_size=batch_size,
                                shuffle=False,
                                dataset_dir=dataset_dir,
                                color_model=color_model)

dataset_train_size = len(dataloaders['train'].dataset)
dataset_test_size = len(dataloaders['test'].dataset)
print("-")

tile_size = 20
magnification = 0.625

threshold_prob = 0.50
threshold_itc = 200 / (0.243 * pow(2, 5))

wsi_images_dir_normal = "{}/testing/normal/wsi".format(dataset_dir)
wsi_images_dir_tumor = "{}/testing/tumor/wsi".format(dataset_dir)

#trained_model_version = "ORCA_512x512__Size-512x512_Epoch-400_Images-100_Batch-1__random_9_operations_all"
#trained_model_version = '022-ORCA512-BCELoss-random9'
#trained_model_version = '023-ORCA512-BCELoss-random8'
#trained_model_version = '024-ORCA512-BCELoss-random9'
#trained_model_version = '025-ORCA512-BCELoss-random8'
#trained_model_version = '026-ORCA512-L1Loss-random9'
trained_model_version = '027-ORCA512-L1Loss-random8'

results_dir = "{}/results/{}/testing".format(dataset_dir_results, trained_model_version)
csv_file_path = "{}/quantitative_analysis_{}.csv".format(results_dir, threshold_prob)

wsi_tissue_patches = {}
with open(csv_file_path, mode='w') as medidas_file:
    medidas_writer = csv.writer(medidas_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    medidas_writer.writerow(['wsi_image', 'patch_image', 'class', 'auc', 'accuracy', 'precision', 'f1/dice', 'jaccard',
                             'sensitivity/recall', 'specificity', 'pixels', 'tp', 'tn', 'fp', 'fn'])

    for batch_idx, (data, target, fname, original_size) in enumerate(dataloaders['test']):

        # load the mask image
        mask_np_img = target[0].numpy()

        # roi x non_roi classes
        wsi_class = "tumor"
        patch_class = "roi"

        # load the predicted image result
        patch_results_dir = "{}/{}/patch/{}x{}/{}".format(results_dir, wsi_class, patch_size[0], patch_size[1],
                                                          fname[0])
        unet_result_img = "{}/01-unet_result/{}".format(patch_results_dir, fname[0])
        predicted_pil_img = load_pil_image(unet_result_img, gray=True) if os.path.isfile(
            unet_result_img) else Image.fromarray(np.zeros(mask_np_img.shape))
        predicted_np_img = np.copy(pil_to_np(predicted_pil_img))
        predicted_np_img = predicted_np_img * (1.0 / 255)
        predicted_np_img = basic_threshold(predicted_np_img, threshold=threshold_prob, output_type="uint8")

        predicted_labels = measure.label(predicted_np_img, connectivity=2)
        predicted_np_img = np.zeros((predicted_np_img.shape[0], predicted_np_img.shape[1]))
        labels = np.unique(predicted_labels)
        properties = measure.regionprops(predicted_labels)
        for lbl in range(1, np.max(labels)):
            major_axis_length = properties[lbl - 1].major_axis_length
            if major_axis_length > threshold_itc:
                predicted_np_img[predicted_labels == lbl] = 1

        # metrics
        auc = roc_auc_score(mask_np_img, predicted_np_img)
        precision = precision_score(mask_np_img, predicted_np_img)
        recall = recall_score(mask_np_img, predicted_np_img)
        accuracy = accuracy_score(mask_np_img, predicted_np_img)
        f1 = f1_score(mask_np_img, predicted_np_img)
        specificity = specificity_score(mask_np_img, predicted_np_img)
        jaccard = jaccard_score(mask_np_img, predicted_np_img)

        total_pixels, tn, fp, fn, tp = tn_fp_fn_tp(mask_np_img, predicted_np_img)

        print(fname[0])
        print("Results for {:26} ({:7} - {:8} - {:04.2f} accuracy)".format(fname[0], patch_class, "unet", accuracy))
        # print("   Precision: \t{}".format(precision))
        # print("   Recall/Sen: \t{}".format(recall))
        # print("   F1/Dice: \t{}".format(f1))
        # print("   Accuracy: \t{}".format(accuracy))
        # print("   Specificity: {}".format(specificity))
        # print("   Jaccard: \t{}".format(jaccard))
        # print("   TP = {} TN = {} FP = {} FN = {}".format(tp, tn, fp, fn))
        # print("-")

        medidas_writer.writerow(
            [fname[0], '-', patch_class, auc, accuracy, precision, f1, jaccard, recall, specificity, total_pixels, tp,
             tn, fp, fn])
