# -*- coding: utf-8 -*-

import imageio
import numpy as np
from PIL import Image
import os
import pathlib
import os

# HCI
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

# 13_13 column_row ordering
# 000_000 001_000 002_000 003_000 ....
# 000_001 001_001 002_001 003_001 ...
# .....
# 000_012 001_012 002_012 003_012 ....

def make_input_image(image, 
                  image_h = 512, 
                  image_w = 512, 
                  RGB = [0.299, 0.587, 0.114]):
    
    traindata_tmp = np.zeros((1, image_h, image_w, 1),
                             dtype=np.float32)

    # tmp = np.float32(
    #     imageio.imread(image))

    tmp = np.float32(
        Image.open(image))
    # tmp = np.float32(imageio.imread(image_path + '/%.2d.png' % (seq + 1)))

    traindata_tmp[0, :, :,
                    0] = (RGB[0] * tmp[:, :, 0] + RGB[1] * tmp[:, :, 1] +
                        RGB[2] * tmp[:, :, 2]) / 255
    
    return traindata_tmp


def make_input(image_path, image_h, image_w, naming = 'HCI', starting = [0,0], 
               image_format = 'png', match_dict = {}, input_angle_resolution = 13, 
               temp_folder = []):
    '''
    data from http://hci-lightfield.iwr.uni-heidelberg.de/
    Sample images ex: Cam000~ Cam080.png  
    '''
    

    output_list = []

    if naming == 'HCI':

        #check first perspective
        input_image = os.path.join(image_path,
                                   'input_Cam{0:03d}.{1:s}'.format(0, image_format))
        if os.path.isfile(input_image):
            tmp = imageio.imread(input_image)

            if len(tmp.shape) == 2:
                image_h, image_w = tmp.shape
            elif len(tmp.shape) == 3:
                image_h, image_w, _ = tmp.shape
            else:
                raise SystemExit('Unexpected image shape {0:d}'.format(len(tmp.shape)))

        for idx in range(81):
            input_image = os.path.join(image_path,
                                       'input_Cam{0:03d}.{1:s}'.format(idx, image_format))
            if os.path.isfile(input_image):
                A = make_input_image(input_image, image_h=image_h, image_w=image_w)
                output_list.append(A)
            else:
                raise SystemExit('Could not find SAI ({0:s})'.format(input_image))
    
    elif naming == 'column_row':

        #check first perspective
        input_image = os.path.join(image_path,
                                   '{0:03d}_{0:03d}.{1:s}'.format(0, image_format))
        
        if temp_folder:
            os.system(f'convert {input_image} -depth 8 {os.path.join(temp_folder,"output.png")}')
            input_image = os.path.join(temp_folder,"output.png")

        if os.path.isfile(input_image):
            tmp = imageio.imread(input_image)

            if len(tmp.shape) == 2:
                image_h, image_w = tmp.shape
            elif len(tmp.shape) == 3:
                image_h, image_w, _ = tmp.shape
            else:
                raise SystemExit('Unexpected image shape {0:d}'.format(len(tmp.shape)))

        input_image = os.path.join(image_path,
                                   '{0:03d}_{1:03d}.{2:s}'.format(starting[0]+8, 
                                                                           starting[1]+8, 
                                                                           image_format))
        
        if not os.path.isfile(input_image):
            starting = [0,0]
            print('{} does not exist'.format(input_image))


        input_image = os.path.join(image_path,
                                   '{0:03d}_{1:03d}.{2:s}'.format(starting[0]+8, 
                                                                           starting[1]+8, 
                                                                           image_format))
        
        if temp_folder:
            os.system(f'convert {input_image} -depth 8 {os.path.join(temp_folder,"output.png")}')
            input_image = os.path.join(temp_folder,"output.png")
        
        if not os.path.isfile(input_image):
            raise SystemExit('Could not find SAI ({0:s}), hence cannot run'.format(input_image))

        for row in range(9):
            for column in range(9):
                input_image = os.path.join(image_path,
                                           '{0:03d}_{1:03d}.{2:s}'.format(starting[0]+column, 
                                                                           starting[1]+row, 
                                                                           image_format))
                if temp_folder:
                    os.system(f'convert {input_image} -depth 8 {os.path.join(temp_folder,"output.png")}')
                    input_image = os.path.join(temp_folder,"output.png")

                if os.path.isfile(input_image):
                    A = make_input_image(input_image, image_h=image_h, image_w=image_w)
                    output_list.append(A)
                else:
                    raise SystemExit('Could not find SAI ({0:s})'.format(input_image))
                
    elif naming == 'ALVC':

        #check first perspective
        input_image = os.path.join(image_path,
                                   'f{0:03d}.{1:s}'.format(1, image_format))
        if os.path.isfile(input_image):
            tmp = imageio.imread(input_image)

            if len(tmp.shape) == 2:
                image_h, image_w = tmp.shape
            elif len(tmp.shape) == 3:
                image_h, image_w, _ = tmp.shape
            else:
                raise SystemExit('Unexpected image shape {0:d}'.format(len(tmp.shape)))
            
        if not match_dict:

            match_dict = gen_row_column(input_angle_resolution)

        test_img = match_dict['{0:03d}_{1:03d}'.format(starting[0]+8, starting[1]+8)]

        input_image = os.path.join(image_path,
                                   '{0:s}.{1:s}'.format(test_img, image_format))
        if not os.path.isfile(input_image):
            print('{} does not exist'.format(input_image))
            raise SystemExit('Could not find SAI ({0:s}), hence cannot run'.format(input_image))

        for row in range(9):
            for column in range(9):
                test_img = match_dict['{0:03d}_{1:03d}'.format(starting[0]+column, 
                                                               starting[1]+row)]
                input_image = os.path.join(image_path,
                                           '{0:s}.{1:s}'.format(test_img,
                                                                image_format))
                if os.path.isfile(input_image):
                    A = make_input_image(input_image, image_h=image_h, image_w=image_w)
                    output_list.append(A)
                else:
                    raise SystemExit('Could not find SAI ({0:s})'.format(input_image))
            

    return output_list

def gen_row_column(angle_resolution, naming = 'f'):

    rows, cols = np.meshgrid(np.arange(angle_resolution), np.arange(angle_resolution))

    names_list = [f"{r:03d}_{c:03d}" for r, c in zip(rows.flatten(), cols.flatten())]

    names = np.reshape(names_list, (angle_resolution, angle_resolution))

    name_order = gen_spiral(names)

    match_dict = {}

    for name_id, name in enumerate(name_order):

        match_dict[name] = 'f{0:03d}'.format(name_id+1)

    return match_dict



def gen_spiral(names):

    m, n = np.shape(names)

    row = int(np.ceil(m/2))-1
    column = int(np.ceil(n/2))-1

    dir = 0
    step = 1
    name_order = []
    name_order.append(str(names[row][column]))

    while step < m:

        if step == (m-1):
            l_max = 4
        else:
            l_max = 3

        for _ in range(1,l_max):

            for _ in range(1,step+1):

                if dir == 0:

                    column -= 1
                
                elif dir == 1:

                    row -= 1

                elif dir == 2:

                    column += 1
                
                elif dir == 3:

                    row += 1

                name_order.append(str(names[row, column]))

            dir += 1

            if dir == 4:
                dir = 0

            
        step += 1
    
    return name_order


# def make_epiinput(image_path, 
#                   seq1, 
#                   image_h = 512, 
#                   image_w = 512, 
#                   view_n = [0], 
#                   RGB = [0.299, 0.587, 0.114]):
    
#     traindata_tmp = np.zeros((1, image_h, image_w, len(view_n)),
#                              dtype=np.float32)
#     i = 0
#     if (len(image_path) == 1):
#         image_path = image_path[0]

#     for seq in seq1:
#         tmp = np.float32(
#             imageio.imread(image_path + '/input_Cam0%.2d.png' % seq))
#         # tmp = np.float32(imageio.imread(image_path + '/%.2d.png' % (seq + 1)))

#         traindata_tmp[0, :, :,
#                       i] = (RGB[0] * tmp[:, :, 0] + RGB[1] * tmp[:, :, 1] +
#                             RGB[2] * tmp[:, :, 2]) / 255
#         i += 1
#     return traindata_tmp


# def make_input(image_path, image_h, image_w, naming = 'HCI'):
#     '''
#     data from http://hci-lightfield.iwr.uni-heidelberg.de/
#     Sample images ex: Cam000~ Cam080.png  
#     '''


#     output_list = []
#     for i in range(81):
#         if (image_path[:11] == 'hci_dataset'):
#             A = make_epiinput(image_path, [i], image_h, image_w, [0], RGB)
#         # print(A.shape)
#         output_list.append(A)

#     return output_list