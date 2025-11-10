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
    '2200_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_horizontal_flip',
    '2201_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_vertical_flip',
    '2202_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_rotation',
    '2203_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_transpose',
    '2204_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_elastic_transformation',
    '2205_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_grid_distortion',
    '2206_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_optical_distortion',
    '2207_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_color_transfer',
    '2208_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_inpainting',
    '2209_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_CLAHE',
    '2210_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_Downscale',
    '2211_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_Equalize',
    '2212_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_HueSaturationValue',
    '2213_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_ISONoise',
    '2214_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_MultiplicativeNoise',
    '2215_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_RandomGravel',
    '2216_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_RingingOvershoot',
    '2217_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_Sharpen',
    '2218_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_Blur',
    '2219_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_Defocus',
    '2220_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_GaussianBlur',
    '2221_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_GlassBlur',
    '2222_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_MedianBlur',
    '2223_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_MotionBlur',
    '2224_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_ZoomBlur',
    '2225_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_Morphological',
    '2226_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_PixelDropout',
    '2227_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_Rotate',
    '2228_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_SafeRotate',
    '2229_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_Perspective',
    '2230_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_ShiftScaleRotate',
    '2231_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_AdvancedBlur',
    '2232_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_ChannelDropout',
    '2233_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_ChannelShuffle',
    '2234_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_ChromaticAberration',
    '2235_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_ColorJitter',
    '2236_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_Emboss',
    '2237_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_FancyPCA',
    '2238_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_GaussNoise',
    '2239_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_ImageCompression',
    '2240_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_InvertImg',
    '2241_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_PlanckianJitter',
    '2242_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_Posterize',
    '2243_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_RGBShift',
    '2244_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_RandomBrightnessContrast',
    '2245_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_RandomFog',
    '2246_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_RandomGamma',
    '2247_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_RandomRain',
    '2248_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_RandomShadow',
    '2249_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_RandomSnow',
    '2250_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_RandomSunFlare',
    '2251_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_RandomToneCurve',
    '2252_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_Solarize',
    '2253_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_Spatter',
    '2254_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_Superpixels',
    '2255_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_ToGray',
    '2256_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_ToSepia',
    '2257_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_UnsharpMask',
    '2258_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_Affine',
    '2259_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_CoarseDropout',
    '2260_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_D4',
    '2261_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_GridDropout',
    '2262_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_Lambda',
    '2263_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_PiecewiseAffine',
    '2264_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_RandomGridShuffle',
    '2265_ORCA512_512x512_Epoch-400_Images-100_Batch-1_L1Loss_Adagrad_XYMasking',
    '2266_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_horizontal_flip',
    '2267_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_vertical_flip',
    '2268_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_rotation',
    '2269_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_transpose',
    '2270_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_elastic_transformation',
    '2271_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_grid_distortion',
    '2272_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_optical_distortion',
    '2273_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_color_transfer',
    '2274_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_inpainting',
    '2275_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_CLAHE',
    '2276_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_Downscale',
    '2277_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_Equalize',
    '2278_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_HueSaturationValue',
    '2279_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_ISONoise',
    '2280_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_MultiplicativeNoise',
    '2281_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_RandomGravel',
    '2282_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_RingingOvershoot',
    '2283_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_Sharpen',
    '2284_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_Blur',
    '2285_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_Defocus',
    '2286_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_GaussianBlur',
    '2287_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_GlassBlur',
    '2288_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_MedianBlur',
    '2289_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_MotionBlur',
    '2290_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_ZoomBlur',
    '2291_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_Morphological',
    '2292_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_PixelDropout',
    '2293_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_Rotate',
    '2294_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_SafeRotate',
    '2295_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_Perspective',
    '2296_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_ShiftScaleRotate',
    '2297_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_AdvancedBlur',
    '2298_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_ChannelDropout',
    '2299_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_ChannelShuffle',
    '2300_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_ChromaticAberration',
    '2301_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_ColorJitter',
    '2302_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_Emboss',
    '2303_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_FancyPCA',
    '2304_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_GaussNoise',
    '2305_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_ImageCompression',
    '2306_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_InvertImg',
    '2307_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_PlanckianJitter',
    '2308_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_Posterize',
    '2309_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_RGBShift',
    '2310_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_RandomBrightnessContrast',
    '2311_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_RandomFog',
    '2312_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_RandomGamma',
    '2313_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_RandomRain',
    '2314_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_RandomShadow',
    '2315_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_RandomSnow',
    '2316_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_RandomSunFlare',
    '2317_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_RandomToneCurve',
    '2318_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_Solarize',
    '2319_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_Spatter',
    '2320_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_Superpixels',
    '2321_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_ToGray',
    '2322_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_ToSepia',
    '2323_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_UnsharpMask',
    '2324_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_Affine',
    '2325_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_CoarseDropout',
    '2326_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_D4',
    '2327_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_GridDropout',
    '2328_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_Lambda',
    '2329_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_PiecewiseAffine',
    '2330_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_RandomGridShuffle',
    '2331_ORCA512_512x512_Epoch-400_Images-100_Batch-1_MSELoss_Adagrad_XYMasking',
    '2332_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_horizontal_flip',
    '2333_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_vertical_flip',
    '2334_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_rotation',
    '2335_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_transpose',
    '2336_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_elastic_transformation',
    '2337_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_grid_distortion',
    '2338_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_optical_distortion',
    '2339_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_color_transfer',
    '2340_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_inpainting',
    '2341_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_CLAHE',
    '2342_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_Downscale',
    '2343_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_Equalize',
    '2344_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_HueSaturationValue',
    '2345_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_ISONoise',
    '2346_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_MultiplicativeNoise',
    '2347_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_RandomGravel',
    '2348_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_RingingOvershoot',
    '2349_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_Sharpen',
    '2350_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_Blur',
    '2351_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_Defocus',
    '2352_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_GaussianBlur',
    '2353_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_GlassBlur',
    '2354_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_MedianBlur',
    '2355_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_MotionBlur',
    '2356_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_ZoomBlur',
    '2357_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_Morphological',
    '2358_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_PixelDropout',
    '2359_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_Rotate',
    '2360_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_SafeRotate',
    '2361_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_Perspective',
    '2362_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_ShiftScaleRotate',
    '2363_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_AdvancedBlur',
    '2364_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_ChannelDropout',
    '2365_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_ChannelShuffle',
    '2366_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_ChromaticAberration',
    '2367_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_ColorJitter',
    '2368_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_Emboss',
    '2369_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_FancyPCA',
    '2370_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_GaussNoise',
    '2371_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_ImageCompression',
    '2372_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_InvertImg',
    '2373_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_PlanckianJitter',
    '2374_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_Posterize',
    '2375_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_RGBShift',
    '2376_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_RandomBrightnessContrast',
    '2377_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_RandomFog',
    '2378_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_RandomGamma',
    '2379_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_RandomRain',
    '2380_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_RandomShadow',
    '2381_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_RandomSnow',
    '2382_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_RandomSunFlare',
    '2383_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_RandomToneCurve',
    '2384_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_Solarize',
    '2385_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_Spatter',
    '2386_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_Superpixels',
    '2387_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_ToGray',
    '2388_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_ToSepia',
    '2389_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_UnsharpMask',
    '2390_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_Affine',
    '2391_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_CoarseDropout',
    '2392_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_D4',
    '2393_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_GridDropout',
    '2394_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_Lambda',
    '2395_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_PiecewiseAffine',
    '2396_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_RandomGridShuffle',
    '2397_ORCA512_512x512_Epoch-400_Images-100_Batch-1_HuberLoss_Adagrad_XYMasking',
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
