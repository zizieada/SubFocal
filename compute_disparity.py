# -*- coding: utf-8 -*-
''' 
The order of LF image files may be different with this file.
(Top to Bottom, Left to Right, and so on..)

If you use different LF images, 

you should change our 'func_makeinput.py' file.

# Light field images: input_Cam000-080.png
# All viewpoints = 9x9(81)

# -- LF viewpoint ordering --
# 00 01 02 03 04 05 06 07 08
# 09 10 11 12 13 14 15 16 17
# 18 19 20 21 22 23 24 25 26
# 27 28 29 30 31 32 33 34 35
# 36 37 38 39 40 41 42 43 44
# 45 46 47 48 49 50 51 52 53
# 54 55 56 57 58 59 60 61 62
# 63 64 65 66 67 68 69 70 71
# 72 73 74 75 76 77 78 79 80

'''

import numpy as np
import os
import time
import argparse
import tensorflow as tf
from LF_func.func_pfm import write_pfm, read_pfm
# from LF_func.func_makeinput import make_epiinput
from LF_func.func_makeinput import make_input
from LF_func.func_model_sub_js import define_SubFocal

# import matplotlib.pyplot as plt
# import cv2
import imageio

parser = argparse.ArgumentParser()
parser.add_argument('-i',
                    '--dataset',
                    default='/input_image',
                    type=str,
                    help='Specify the path to the input folder.')
parser.add_argument('--imagew',
                    type=int,
                    default=512,
                    help='Define width of one SAI.')
parser.add_argument('--imageh',
                    type=int,
                    default=512,
                    help='Define height of one SAI.')
parser.add_argument('--angularviews',
                    type=int,
                    default=8,
                    help='Define the number of rows/columns (needs to be a square)')
parser.add_argument('--format',
                    default='png',
                    type=str,
                    help='Specify the format of the input images.')
parser.add_argument('--outputpath',
                    default='',
                    type=str,
                    help='Specify the output path. If none supplied, default will be used.')
parser.add_argument('--outputname',
                    default='',
                    type=str,
                    help='Specify the name of the output. If none supplied, default will be used.')
parser.add_argument('--naming',
                    default='HCI',
                    type=str,
                    help='Either HCI or column_row')
parser.add_argument(
    '--starting',
    nargs='+',
    default=[2, 2],
    type=int,
    help='Define starting coordinates for the column_row method.).'
)

if __name__ == '__main__':

    # parse input arguments
    args = parser.parse_args()

# Check if GPU is available
    if tf.test.is_gpu_available():
        print("GPU is available.")
        # Limit GPU memory growth
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            # tf.config.experimental.set_virtual_device_configuration(
            #     device,
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15360)])  # Limit to 15GB
            
    else:
        print("GPU is not available. Using CPU.")
        # physical_devices = tf.config.list_physical_devices('CPU')
        # for device in physical_devices:
        #     tf.config.experimental.set_virtual_device_configuration(
        #         device,
        #         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])  # Limit to 1GB
        # tf.config.experimental.set_memory_limit(24576)  # Limit to 24GB

        # os.environ["OMP_NUM_THREADS"] = num_threads
        # config = tf.ConfigProto()
        # config.intra_op_parallelism_threads = num_threads
        # config.inter_op_parallelism_threads = num_threads
        # tf.Session(config=config)

    # Input : input_Cam000-080.png
    # Depth output : image_name.pfm

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.outputpath == '':
        dir_output = os.path.join(script_dir,'output_disparity')#'SubFocal_sub0.5_js_0.1_e10_export'
    else:
        dir_output = args.outputpath


    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    # GPU setting ( rtx 3090 - gpu0 )
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    '''
    /// Setting 1. LF Images Directory

    LFdir = 'synthetic': Test synthetic LF images (from 4D Light Field Benchmark)
                                   "A Dataset and Evaluation Methodology for 
                                   Depth Estimation on 4D Light Fields".
                                   http://hci-lightfield.iwr.uni-heidelberg.de/

    '''
    # LFdir = 'synthetic'

    # if (LFdir == 'synthetic'):
    #     dir_LFimages = [
    #         'hci_dataset/stratified/backgammon', 'hci_dataset/stratified/dots',
    #         'hci_dataset/stratified/pyramids',
    #         'hci_dataset/stratified/stripes', 'hci_dataset/training/boxes',
    #         'hci_dataset/training/cotton', 'hci_dataset/training/dino',
    #         'hci_dataset/training/sideboard'
    #     ]

    #     image_w = 512
    #     image_h = 512

    image_path = args.dataset

    image_w = args.imagew
    image_h = args.imageh
        
    # dir_LFimages = []

    # number of views ( 0~8 for 9x9 )
    # AngualrViews = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    AngualrViews = [x for x in range(args.angularviews + 1)]

    path_weight = os.path.join(script_dir,'LF_checkpoint','SubFocal_sub_0.5_js_0.1_ckp','iter0010_valmse0.768_bp1.93.hdf5')
    # path_weight = '/root/SubFocal/LF_checkpoint/SubFocal_sub_0.5_js_0.1_ckp/iter0010_valmse0.768_bp1.93.hdf5'

    img_scale = 1  # 1 for small_baseline(default) <3.5px,
    # 0.5 for large_baseline images   <  7px

    img_scale_inv = int(1 / img_scale)
    ''' Define Model ( set parameters )'''

    model_learning_rate = 0.001
    model_512 = define_SubFocal(round(img_scale * image_h),
                                round(img_scale * image_w), AngualrViews,
                                model_learning_rate)
    ''' Model Initialization '''

    model_512.load_weights(path_weight)
    # dum_sz = model_512.input_shape[0]
    # dum = np.zeros((1, dum_sz[1], dum_sz[2], dum_sz[3]), dtype=np.float32)
    # tmp_list = []
    # for i in range(81):
    #     tmp_list.append(dum)
    # dummy = model_512.predict(tmp_list, batch_size=1)

    avg_attention = []
    """  Depth Estimation  """
    # for image_path in dir_LFimages:

    if args.naming not in ['HCI', 'column_row', 'ALVC']:
        naming = 'HCI'
    else:
        naming = args.naming

    if naming in ['column_row', 'ALVC']:
        starting = args.starting
    else:
        starting = [2,2]

    start = time.time()

    #val_list = make_input(image_path, image_h, image_w, AngualrViews)
    val_list = make_input(image_path, image_h, image_w, 
                          naming = naming, starting = starting, #[0,0], 
                          image_format = str(args.format))

    input_prep_time = time.time()

    print("runtime: %.5f(s)" % (input_prep_time - start))

    # predict
    val_output_tmp, _ = model_512.predict(val_list, batch_size=1)

    runtime = time.time() - start
    print("runtime: %.5f(s)" % runtime)

    if args.outputname == '':
        output_name = image_path.split('/')[-1]
    else:
        output_name = args.outputname

    # save .pfm file
    output_jpg = os.path.join(dir_output, '{0:s}.jpg'.format(output_name))
    output_pfm = os.path.join(dir_output, '{0:s}.pfm'.format(output_name))
    imageio.imsave(output_jpg,
                    val_output_tmp[0, :, :])
    write_pfm(val_output_tmp[0, :, :],
                output_pfm)
    print('pfm file saved as %s' % output_pfm)
