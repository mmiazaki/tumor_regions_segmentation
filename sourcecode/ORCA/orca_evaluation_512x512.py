import os
import sys
import time

current_path = os.path.abspath('.')
root_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(root_path)

from sourcecode.Utils.oscc_dataloader import *
from sourcecode.Utils.orca_load_dataset_512x512 import *
from sourcecode.Utils.evaluation_utils import *

from datetime import timedelta, datetime

dataset_dir = "../../datasets/ORCA_512x512"
dataset_dir_results = "../../datasets/ORCA_512x512"
csv_output = dataset_dir+'/ORCA_512x512_final_avg_measures.csv'

batch_size = 1
patch_size = (512, 512)
color_model = "LAB"

tile_size = 20
magnification = 0.625
threshold_prob = 0.50
threshold_itc = 200 / (0.243 * pow(2, 5))

# loads our trained fcn model
load_models = [
    # '2474_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adagrad_Downscale',
    ]


start_time = start_model_time = time.time()

load_models_len = len(load_models)
for i, trained_model_version in enumerate(load_models):
    print(':::::: {} ::::::'.format(trained_model_version))

    dataloaders = create_dataloader(samples_function=orca512_load_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    dataset_dir=dataset_dir,
                                    color_model=color_model)

    dataset_train_size = len(dataloaders['train'].dataset)
    dataset_test_size = len(dataloaders['test'].dataset)

    results_dir = "{}/results/{}/testing".format(dataset_dir_results, trained_model_version)
    csv_file_path = "{}/quantitative_analysis_{}.csv".format(results_dir, threshold_prob)

    wsi_tissue_patches = {}
    with open(csv_file_path, mode='w') as medidas_file:
        medidas_writer = csv.writer(medidas_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        medidas_writer.writerow(['wsi_image', 'patch_image', 'class', 'auc', 'accuracy', 'precision', 'f1/dice', 'jaccard',
                                 'sensitivity/recall', 'specificity', 'pixels', 'tp', 'tn', 'fp', 'fn'])

        dataloaders_len = len(dataloaders['test'])
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

            # print(fname[0])
            print("(Models: {}/{} - Images: {}/{}) {:26} ({:7} - {:8} - {:04.2f} accuracy)".format(i+1, load_models_len, batch_idx+1, dataloaders_len, fname[0], patch_class, "unet", accuracy))
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

    calculate_avg_results(csv_file_path, csv_output, trained_model_version)

    print('=====================')
    now = time.time()
    elapsed_model_time = now - start_model_time
    total_elapsed_time = now - start_time
    print('Total elapsed time: {}'.format(timedelta(seconds=total_elapsed_time)))
    print('Elapsed time for model {}: {}'.format(trained_model_version[:4], timedelta(seconds=elapsed_model_time)))
    mean_model_time = total_elapsed_time / (i + 1)
    estimated_time = mean_model_time * (load_models_len - i - 1)
    print('Mean model time: {}'.format(timedelta(seconds=mean_model_time)))
    print('Estimated finish time: {}'.format(timedelta(seconds=estimated_time)))
    print('Estimated finish date: {}'.format(datetime.fromtimestamp(now + estimated_time).strftime('%d/%m/%Y %H:%M:%S')))
    start_model_time = now
    print('=====================')
