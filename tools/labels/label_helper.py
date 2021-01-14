import cv2
import os
import cv2

def crop_img_for_lable2( fn ):
    img = cv2.imread(fn)
    img_crop =  img[ int(0.35/1.35 * 576): , :, : ]
    cv2.imwrite( './input/crop_new.jpg', img_crop )


def crop_img_for_lable( fn, save_path=None ):
    img = cv2.imread(fn)

    height_resize = int(576*1.35)
    
    img_copy = cv2.resize( img, (1024, height_resize) )
    img_crop = cv2.resize( img_copy[ height_resize-576: , :, : ], (1280, 720))

    if save_path is not None:
        cv2.imwrite( save_path, img_crop )

def crop_img_batch(folder = './input/', output='./panos_crop'): 
    for folderName, subfolders, filenames in os.walk(folder):
        for filename in filenames:
            crop_img_for_lable( os.path.join( folderName, filename ), os.path.join( output, filename ) )

crop_img_batch( './input', './panos_crop' )



