import os
import airsim
import cv2
import numpy as np
import pprint
from depth_2_distance import (meshgrid_from_img, depth_2_distance)

def scale_float_array(array, maxA=50):
    '''array (NumPy): The float image.
    maxA (float): Upper limit of array for clipping.'''
    minA = array.min()
    assert(minA < maxA)
    return (array - minA) / ( maxA - minA )

def record(client, record_dir, weather_record, z_distance, id, vehicle_name = ''):
    pp = pprint.PrettyPrinter(indent=4)
    # 0为前向中间 1为前向右侧 2为前向左侧 3为底部中间 4为后向中间
    # 但就目前观测的结果，0为前方仰视，1为前方平视，为2为前方俯视，3 4正确
    request = [
    # airsim.ImageRequest('0', airsim.ImageType.Scene, pixels_as_float=False, compress=True),
    airsim.ImageRequest('cube_LT', airsim.ImageType.CubeScene, pixels_as_float=False, compress=True),
    # airsim.ImageRequest('cube_LT', airsim.ImageType.CubeDepth, pixels_as_float=True,  compress=False),

    airsim.ImageRequest('cube_RT', airsim.ImageType.CubeScene, pixels_as_float=False, compress=True),
    # airsim.ImageRequest('cube_RT', airsim.ImageType.CubeDepth, pixels_as_float=True,  compress=False),
    
    airsim.ImageRequest('cube_LB', airsim.ImageType.CubeScene, pixels_as_float=False, compress=True),
    # airsim.ImageRequest('cube_LB', airsim.ImageType.CubeDepth, pixels_as_float=True,  compress=False),

    airsim.ImageRequest('cube_RB', airsim.ImageType.CubeScene, pixels_as_float=False, compress=True),
    # airsim.ImageRequest('cube_RB', airsim.ImageType.CubeDepth, pixels_as_float=True,  compress=False),

    airsim.ImageRequest('cube_Center', airsim.ImageType.CubeScene, pixels_as_float=False, compress=True),
    airsim.ImageRequest('cube_Center', airsim.ImageType.CubeDepth, pixels_as_float=True,  compress=False),
    ]

    responses = client.simGetImages(request)

    # --- 新增日志记录逻辑 ---
    if not responses:
        print("警告: 未收到任何图像响应，跳过记录。")
        return

    # 1. 定义日志文件路径和表头
    # 将日志文件放在 record_dir 的上一级目录
    parent_dir = os.path.abspath(os.path.join(record_dir, os.pardir))
    os.makedirs(parent_dir, exist_ok=True)
    log_filepath = os.path.join(parent_dir, 'log.csv')
    log_header = 'VehicleName,TimeStamp,POS_X,POS_Y,POS_Z,Q_W,Q_X,Q_Y,Q_Z,ImageFiles\n'

    # 2. 如果日志文件不存在，则创建并写入表头
    if not os.path.exists(log_filepath):
        with open(log_filepath, 'w') as f:
            f.write(log_header)

    # 3. 获取车辆位姿和时间戳 (所有响应共享这些信息)
    pose = client.simGetVehiclePose(vehicle_name)
    timestamp = responses[0].time_stamp
    
    # 4. 准备日志行的数据
    pos = pose.position
    q = pose.orientation
    log_data = [
        vehicle_name,
        timestamp,
        f"{pos.x_val:.4f}", f"{pos.y_val:.4f}", f"{pos.z_val:.4f}",
        f"{q.w_val:.6f}", f"{q.x_val:.6f}", f"{q.y_val:.6f}", f"{q.z_val:.6f}"
    ]


    generated_files = []
    for i, response in enumerate(responses):
        save_path = f"{response.camera_name}_{response.image_type}"
        os.makedirs(os.path.join(record_dir, save_path), exist_ok=True)
        save_name = f"{id:06d}"
        if response.pixels_as_float:
            print( "Type %d, size %d, pos \n%s" % \
                ( response.image_type, 
                  len(response.image_data_float), 
                  pprint.pformat(response.camera_position) ) )
            # Get the raw floating-point data.
            pfm_array = airsim.get_pfm_array(response).reshape((response.height, response.width, 1))

            # Create the meshgrid.
            xx, yy = meshgrid_from_img(pfm_array)

            # Convert the depth values to distance.
            dist_array = np.zeros_like(pfm_array)
            if ( not depth_2_distance(pfm_array, xx, yy, dist_array) ):
                raise Exception('Failed to conver the depth to distance. ')

            # Save the floating-point data as compressed PNG file.
            img_array = dist_array.view('<u1')
            # cv2.imwrite(os.path.normpath(os.path.join(record_dir, '%02d.png' % (i))), img_array)
            np.savez_compressed(os.path.normpath(os.path.join(record_dir, save_path, f'{save_name}.npz')), img_array)

            generated_files.append(save_name + '.npz')

            # Visualize the depth.
            scaled = scale_float_array(dist_array, maxA=50)
            scaled_grey = (np.clip( scaled, 0, 1 ) * 255).astype(np.uint8)
            cv2.imwrite(os.path.normpath(os.path.join(record_dir, save_path, f'{save_name}_vis.png')), scaled_grey)

        else:
            print("Type %d, size %d, pos \n%s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
            
            # Decode the image directly from the bytes.
            decoded = cv2.imdecode(np.frombuffer(response.image_data_uint8, np.uint8), -1)

            # Write the image as a PNG file.
            # The incoming data has 4 channels. The alpha channel is all zero.
            cv2.imwrite(os.path.normpath(os.path.join(record_dir, save_path, f'{save_name}.png')), decoded[:, :, :3])
            generated_files.append(save_name + '.png')

    # 5. 将收集到的文件名列表合并为一个字符串，并追加到日志数据中
    image_files_str = ";".join(generated_files)
    log_data.append(image_files_str)

    # 6. 将完整的日志行写入文件
    with open(log_filepath, 'a') as f:
        f.write(",".join(map(str, log_data)) + '\n')
    
    print(f"成功记录元数据到: {log_filepath}")

def make_record_dir(path):
    os.makedirs(path, exist_ok=True)

