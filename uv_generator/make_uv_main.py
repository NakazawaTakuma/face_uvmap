import cv2
import numpy as np

import math
from matplotlib import path
from . import define_data
import time

face_index_all = np.array([10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338])      



#landmark position
def face_area(index_list, face_landmarks):
    position_list = np.empty(index_list)
    for i in  range(len(index_list)): 
        position_list[i] = [face_landmarks.landmark[index_list[i]].x,face_landmarks.landmark[index_list[i]].y]

    return position_list 





def generate_uvmap(image,face_landmarks,generate_image_size):

    
    start = time.time()
    face_lposi = np.array([[face_landmarks.landmark[a].x,face_landmarks.landmark[a].y] for a in range(468)])
    face_lposi = face_lposi[:, [0, 1]]
    triangle_index = define_data.triangle_index
    uv_xy = define_data.uv_xy

    
  
    # image shape
    imgh, imgw = image.shape[:2]

    #create pix array


    triangle_int_posi_list=np.full(len(triangle_index),None)

    ref_posi_list=np.full(len(triangle_index),None)

    start = time.time()
    q = 0
    for i in  range(len(triangle_index)):

        
        index = np.array(triangle_index[i])
        point = np.zeros( (len(index), 2))
 
    
        for j in range(len(index)):
            point[j][0] = imgw*face_landmarks.landmark[index[j]].x # x
            point[j][1] = imgh*face_landmarks.landmark[index[j]].y # y

        
        #affine

        dst_pts = np.array([[point[0][0], point[0][1]], [point[1][0], point[1][1]], [point[2][0], point[2][1]]], dtype=np.float32)
        src_pts = np.array([[uv_xy[index[0]][0]*generate_image_size, uv_xy[index[0]][1]*generate_image_size], [uv_xy[index[1]][0]*generate_image_size, uv_xy[index[1]][1]*generate_image_size], [uv_xy[index[2]][0]*generate_image_size, uv_xy[index[2]][1]*generate_image_size]], dtype=np.float32)

        #Find the integer coordinates in the triangle (triangle_int_posi )
        src_pts_max_posi = np.max(src_pts, axis=0)
        src_pts_min_posi = np.min(src_pts, axis=0)

 


        x_int_posi = np.array(range(int(src_pts_min_posi[0])+1, int(src_pts_max_posi[0])+1))
        y_int_posi = np.array(range(int(src_pts_min_posi[1])+1, int(src_pts_max_posi[1])+1))
        int_posi = np.array([(i,j) for i in x_int_posi for j in y_int_posi])
        
        
        

        triangle_int_posi = []
        for j in range(len(int_posi)):
            polygon = path.Path(src_pts)
            if polygon.contains_point(int_posi[j]):
                triangle_int_posi.append([int_posi[j][0],int_posi[j][1],1])
                q += 1

        if not triangle_int_posi:
            continue





        affin = cv2.getAffineTransform(src_pts, dst_pts)


        da = -math.atan2(affin[1,0],affin[0,0])
        s1 = ( affin[0,0] / math.cos(da) )* ( float(generate_image_size) / float(imgh))
        s2 = ( affin[1,1] / math.cos(da) )* ( float(generate_image_size) / float(imgw))

        if s1 < 0.1 or s2 < 0.1:
            continue

        triangle_int_posi_list[i]=np.array(triangle_int_posi)
        triangle_int_posi = np.array([triangle_int_posi], dtype='float')

 


        affin = affin.tolist()
        affin.append([0,0,1])
        affin = np.asarray(affin, dtype='float')
         


        ref_posi = np.einsum("lk,ijk->ijl",affin,triangle_int_posi)

        ref_posi = ref_posi[...,:2]


        ref_posi = ref_posi[0]


        ref_posi = ref_posi[:, [1,0]]


        ref_posi = ref_posi.tolist()


        ref_posi_list[i] = ref_posi



   
    all_ref_posi = [[[0 for k in range(2)] for j in range(generate_image_size)] for i in range(generate_image_size)]
    
    for i in range(len(triangle_int_posi_list)):

        if triangle_int_posi_list[i] is None:

            continue
        for j in range(len(triangle_int_posi_list[i])):

            all_ref_posi[triangle_int_posi_list[i][j][1]][triangle_int_posi_list[i][j][0]]=ref_posi_list[i][j]

    
    all_ref_posi = np.asarray(all_ref_posi, dtype='float')

    

    #Linear interpolation method
    linear_xy = {}
    linear_xy['upleft'] = all_ref_posi.astype(int)
    linear_xy['upleft'] = np.where(linear_xy['upleft'] > imgw-1, imgw-1, linear_xy['upleft'])

    linear_xy['downleft'] =linear_xy['upleft'] + [1,0]
    linear_xy['downleft'] = np.where(linear_xy['downleft'] > imgh-1, imgh-1, linear_xy['downleft'])

    linear_xy['upright']= linear_xy['upleft'] + [0,1]
    linear_xy['upright'] = np.where(linear_xy['upright'] > imgw-1, imgw-1, linear_xy['upright'])
    
    linear_xy['downright'] = linear_xy['upleft'] + [1,1]
    linear_xy['downright'] = np.where(linear_xy['downright'] > imgw-1, imgw-1, linear_xy['downright'])
    
    
    upleft_diff = all_ref_posi - linear_xy['upleft']

    
    linear_weight = {}
    linear_weight['upleft'] = (1-upleft_diff[...,0])*(1-upleft_diff[...,1])
    linear_weight['downleft'] = upleft_diff[...,0]*(1-upleft_diff[...,1])
    linear_weight['upright'] = (1-upleft_diff[...,0])*upleft_diff[...,1]
    linear_weight['downright'] = upleft_diff[...,0]*upleft_diff[...,1]
    linear_weight['maxweight'] = linear_weight['upleft']
    for direction in linear_xy.keys():
        if direction == 'upleft':
            continue
        linear_weight['maxweight'] = np.fmax(linear_weight['maxweight'],linear_weight[direction])

    img_int = {}


    for direction in linear_xy.keys():
         xy = linear_xy[direction]
         img_int[direction] = image[xy[...,0],xy[...,1]]


    for i in range(len(img_int['upleft'])):
        for j in range(len(img_int['upleft'][i])):
            for direction in linear_xy.keys():
                # if contain (0,0,0) color
               
                if np.all(img_int[direction][i][j] == 0):
                    
                    if linear_weight[direction][i][j] == linear_weight['maxweight'][i][j]: 
                        for direction2 in linear_xy.keys():
                            img_int[direction2][i][j] = [0,0,0]
                    # Delete weight of [0,0,0] and Distributed
                        
                    else:
                        
                        for direction2 in linear_xy.keys():
                            if direction2 == direction or linear_weight[direction][i][j] == 1:
                                continue

                            linear_weight[direction2][i][j] = linear_weight[direction2][i][j]+ (linear_weight[direction2][i][j]/(1.-linear_weight[direction][i][j]))*linear_weight[direction][i][j]
                            

                        linear_weight[direction][i][j] = 0
                
    

    linear_with_weight = {}
    for direction in linear_xy.keys():
        weight = linear_weight[direction]
        linear_with_weight[direction] = np.einsum('ij,ijk->ijk',weight,img_int[direction])
    img_linear = sum(linear_with_weight.values())
    np.set_printoptions(threshold=np.inf)
    return img_linear