import os
import sys
import csv

current_path = os.path.abspath('.')
root_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(root_path)

from skimage import measure
from sourcecode.ORCA.orca_dataloader_512x512 import *
from sourcecode.wsi_image_utils import *
from sourcecode.evaluation_utils import *


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
    # '100_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_25_operations',
    # '101_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_16_operations',
    # '102_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_18_operations',
    # '103_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_24_operations',
    # '104_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_standard_8_operations',
    # '105_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_inpainting_augmentation',
    # '106_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_no_augmentation',
    # '107_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_color_augmentation',
    # '108_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_standard_4_operations',
    # '109_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_standard_2_operations',
    # '110_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_standard_9_operations',
    # '111_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_standard_8_operations',
    # '112_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_2_operations',
    # '113_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_4_operations',
    # '114_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_standard_1_operations',
    # '117_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_1_operations',
    # '115_ORCA_512x512___512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '116_ORCA_512x512___512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '118_ORCA_512x512____Size-512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_1_operations',
    # '121_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_CLAHE',
    # '122_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_Downscale',
    # '123_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_Equalize',
    # '124_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_HueSaturationValue',
    # '125_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_Morphological',
    # '131_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_ISONoise',
    # '132_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_MultiplicativeNoise',
    # '133_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_RandomGravel',
    # '134_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_RingingOvershoot',
    # '135_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_Rotate',
    # '141_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_Sharpen',
    # '142_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_Blur',
    # '144_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_GaussianBlur',
    # '145_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_SafeRotate',
    # '151_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_GlassBlur',
    # '152_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_MedianBlur',
    # '153_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_MotionBlur',
    # '154_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_ZoomBlur',
    # '155_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_Perspective',
    # '161_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_no_augmentation',
    # '162_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_8_operations',
    # '163_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_standard_8_operations',
    # '164_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_standard_8_operations',
    # '165_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_no_augmentation',
    # '171_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_ShiftScaleRotate',
    # '172_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_horizontal_flip',
    # '173_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_vertical_flip',
    # '174_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_rotation',
    # '175_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_transpose',
    # '176_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_grid_distortion',
    # '177_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_optical_distortion',
    # '143_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_Defocus',
    # '181_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_standard_8_operations',
    # '182_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '183_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '184_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '191_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '192_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '193_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '195_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '205_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '215_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '224_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '194_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '201_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '202_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '203_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '204_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '211_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '212_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '213_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '214_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '221_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '222_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '223_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_9_operations',
    # '225_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_ShiftScaleRotate',
    # '226_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_CLAHE',
    # '227_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_Equalize',
    # '228_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_ShiftScaleRotate',
    # '229_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_CLAHE',
    # '230_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_Equalize',
    # '231_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_random_3_operations',
    # '232_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_standard_3_operations',
    # '233_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_random_3_operations',
    # '234_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_standard_3_operations',
    # '235_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_horizontal_flip',
    # '236_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_vertical_flip',
    # '237_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_rotation',
    # '238_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_transpose',
    # '239_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_elastic_transformation',
    # '240_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_grid_distortion',
    # '241_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_optical_distortion',
    # '242_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_color_transfer',
    # '243_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_inpainting',
    # '244_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_Downscale',
    # '245_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_HueSaturationValue',
    # '246_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_ISONoise',
    # '247_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_MultiplicativeNoise',
    # '248_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_RandomGravel',
    # '249_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_RingingOvershoot',
    # '250_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_Sharpen',
    # '251_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_Blur',
    # '252_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_Defocus',
    # '253_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_GaussianBlur',
    # '254_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_GlassBlur',
    # '255_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_MedianBlur',
    # '256_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_MotionBlur',
    # '257_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_ZoomBlur',
    # '258_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_Morphological',
    # '259_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_PixelDropout',
    # '260_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_Rotate',
    # '261_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_SafeRotate',
    # '262_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_Perspective',
    # '263_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_horizontal_flip',
    # '265_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_rotation',
    # '269_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_optical_distortion',
    # '264_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_vertical_flip',
    # '266_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_transpose',
    # '267_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_elastic_transformation',
    # '268_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_grid_distortion',
    # '270_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_color_transfer',
    # '272_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_CLAHE',
    # '273_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_Downscale',
    # '279_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_RingingOvershoot',
    # '280_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_Sharpen',
    # '284_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_GlassBlur',
    # '288_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_Morphological',
    # '300_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_AdvancedBlur',
    # '301_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_ChannelDropout',
    # '302_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_ChannelShuffle',
    # '303_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_ChromaticAberration',
    # '304_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_ColorJitter',
    # '305_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_Emboss',
    # '306_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_FancyPCA',
    # '307_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_GaussNoise',
    # '308_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_ImageCompression',
    # '309_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_InvertImg',
    # '311_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_PlanckianJitter',
    # '312_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_Posterize',
    # '313_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_RGBShift',
    # '314_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_RandomBrightnessContrast',
    # '315_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_RandomFog',
    # '316_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_RandomGamma',
    # '317_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_RandomRain',
    # '318_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_RandomShadow',
    # '320_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_RandomSunFlare',
    # '321_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_RandomToneCurve',
    # '322_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_Solarize',
    # '323_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_Spatter',
    # '324_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_Superpixels',
    # '325_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_ToGray',
    # '326_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_ToSepia',
    # '327_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_UnsharpMask',
    # '328_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_Affine',
    # '329_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_CoarseDropout',
    # '330_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_D4',
    # '332_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_Lambda',
    # '319_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_RandomSnow',
    # '331_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_GridDropout',
    # '338_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_RandomGridShuffle',
    # '336_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_PiecewiseAffine',
    # '341_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_XYMasking',
    # '274_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_Equalize',
    # '275_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_HueSaturationValue',
    # '277_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_MultiplicativeNoise',
    # '278_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_RandomGravel',
    # '281_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_Blur',
    # '282_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_Defocus',
    # '283_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_GaussianBlur',
    # '284_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_GlassBlur',
    # '285_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_MedianBlur',
    # '286_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_MotionBlur',
    # '287_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_ZoomBlur',
    # '289_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_PixelDropout',
    # '290_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_Rotate',
    # '291_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_SafeRotate',
    # '292_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_Perspective',
    # '293_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_ShiftScaleRotate',
    # '345_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_Posterize',
    # '346_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_Posterize',
    # '347_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_3_operations',
    # '348_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_random_3_operations',
    # '349_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_random_3_operations',
    # '350_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_standard_3_operations',
    # '351_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_standard_3_operations',
    # '352_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_standard_3_operations',
    # '353_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_random_11_operations',
    # '354_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_random_11_operations',
    # '355_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_random_11_operations',
    # '356_ORCA512_512x512_Epoch-400_Images-100_Batch-1_BCELoss_Adam_standard_11_operations',
    # '357_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_standard_11_operations',
    # '358_ORCA512_512x512_Epoch-400_Images-100_Batch-1_SmoothL1Loss_Adam_standard_11_operations',
    # '441_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_ShiftScaleRotate',
    # '442_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_AdvancedBlur',
    # '443_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_ChannelDropout',
    # '444_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_ChannelShuffle',
    # '445_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_ChromaticAberration',
    # '446_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_ColorJitter',
    # '447_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_Emboss',
    # '448_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_FancyPCA',
    # '449_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_GaussNoise',
    # '450_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_ImageCompression',
    # '451_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_InvertImg',
    # '452_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_PlanckianJitter',
    # '453_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_Posterize',
    # '454_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_RGBShift',
    # '455_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_RandomBrightnessContrast',
    # '456_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_RandomFog',
    # '457_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_RandomGamma',
    # '458_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_RandomRain',
    # '459_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_RandomShadow',
    # '460_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_RandomSnow',
    # '461_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_RandomSunFlare',
    # '462_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_RandomToneCurve',
    # '463_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_Solarize',
    # '464_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_Spatter',
    # '465_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_Superpixels',
    # '466_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_ToGray',
    # '467_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_ToSepia',
    # '468_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_UnsharpMask',
    # '469_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_Affine',
    # '470_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_CoarseDropout',
    # '471_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_D4',
    # '472_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_GridDropout',
    # '473_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_Lambda',
    # '474_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_PiecewiseAffine',
    # '475_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_RandomGridShuffle',
    # '476_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_XYMasking',
    '271_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_inpainting',
    '276_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adam_ISONoise',
    ]



for trained_model_version in load_models:
    print(':::::: {} ::::::'.format(trained_model_version))

    dataloaders = create_dataloader(batch_size=batch_size,
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
            print("Results for ({}) {:26} ({:7} - {:8} - {:04.2f} accuracy)".format(batch_idx+1, fname[0], patch_class, "unet", accuracy))
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
