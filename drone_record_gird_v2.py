from cmath import pi
from random import randint,uniform,choice
from re import U
from turtle import position
# import setup_path
import airsim

import numpy as np
import os, sys, time, datetime
import threading
sys.path.append(os.path.dirname(__file__))
import tempfile
import pprint
import cv2, math
import matplotlib.pyplot as plt
# from utils import get_state, quaternion_rotation_matrix, distance
# from depth2PointCloud import depth2PointCloud
# from occupancy_grid import Map
# from bresenhan_nd import bresenhamline
# from RRT3d import rrt3d
# import pymap3d as pm3d
from scipy.spatial.transform import Rotation
from airsim.types import GeoPoint
import argparse
# from segmentation import set_segmentation_color
# from bresenhan_nd import bresenhamline
from record_grid_v2 import record, make_record_dir
from sympy import *
from utils_grid import load_config

def get_cur_position(client):
    # n, e, d = client.getMultirotorState().kinematics_estimated.position
    # w_val, x_val, y_val,z_val = client.getMultirotorState().kinematics_estimated.orientation
    n, e, d = client.simGetGroundTruthKinematics().position
    w_val, x_val, y_val,z_val = client.simGetGroundTruthKinematics().orientation
    return (n, e, d)
    
def init_client():
    # connect to the AirSim simulator
    # client = airsim.MultirotorClient()
    # client.confirmConnection()
    # client.enableApiControl(True)
    
    # ComputerVision init
    client = airsim.VehicleClient()
    client.confirmConnection()
    return client
  
def reset_drone(client,x_distance=0,y_distance=0,z_distance=-1,yaw_degree=0) :  
    # client.confirmConnection()
    # client.enableApiControl(True)
    # client.armDisarm(True)
    # ComputerVision init
    # client = airsim.VehicleClient()
    # client.confirmConnection()
    # 设置玩家起始点
    print("set player position")
    VehiclePose=client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x_distance, y_distance, z_distance), airsim.to_quaternion(math.radians(randint(-30,30)), 0, math.radians(yaw_degree))), ignore_collision=True)
    # VehiclePose=client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x_step, y_step, 0), airsim.to_quaternion(0, 0, 0)), ignore_collision=False)
    # VehiclePose=client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(step, step, 0), airsim.to_quaternion(0, 0, 0)), True)   

def write_line(client,drone_ned1,drone_ned2):
    #画一条到新位置的线
    # rgb =[0,0,0,1]
    color=uniform(0,35)/255
    rgb = [color for i in range(3)]
    rgb.append(1)
    thickness =uniform(1.5,4)
    client.simPlotLineList([airsim.Vector3r(drone_ned1[0],drone_ned1[1],drone_ned1[2]), airsim.Vector3r(drone_ned2[0],drone_ned2[1],drone_ned2[2])], 
                            color_rgba=rgb, 
                            thickness = thickness, 
                            duration = 3,
                            is_persistent=False)
    # client.simPlotLineList([airsim.Vector3r(drone_ned1[0],drone_ned1[1],drone_ned1[2]), airsim.Vector3r(drone_ned2[0],drone_ned2[1],drone_ned2[2])], 
    #                         color_rgba=rgb, 
    #                         thickness = thickness, 
    #                         is_persistent=True)
    
# def record_by_Yaw(client, record_dir, z_distance):
def record_by_Yaw(client, record_dir, z_distance,weather_record):
    # try:
    # if "_weather_random" in record_dir:
    #     weather, weather_str, weather_degree = reset_weather(client)
    # else:
    #     weather,weather_str, weather_degree = -1, "defualt", 1
    # weather_record = (weather_str,weather_degree)
    # drivetrain = airsim.DrivetrainType.ForwardOnly
    # 四旋翼的朝向始终与前进方向相差90度，也就是四旋翼始终向左侧方向运动
    # yaw_mode = airsim.YawMode(False, 90)
    # 四旋翼的朝向始终与前进方向相差0度，也就是四旋翼始终向前方方向运动
    # ω = pi*45/180 #角速度
    # yaw_mode = airsim.YawMode(True, 45)
    # # 四旋翼不管速度方向是什么，yaw角以10度/秒的速度旋转
    # drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom
    # yaw_mode = airsim.YawMode(True, 10)
    # record_times = 8
    # client.moveByAngleRatesZAsync(0, 0, 45, 0, record_times)
    # client.moveToPositionAsync(path[i][0], path[i][1], path[i][2], velocity, drivetrain=drivetrain, yaw_mode=yaw_mode)
    # record(client, record_times, record_dir, weather_record)
    record(client, record_dir, weather_record, z_distance)
    # #清除天气，避免叠加
    # client.simSetWeatherParameter(weather, 0.0)
    # except:
    #     errorType, value, traceback = sys.exc_info()
    #     print("record_by_Yaw threw exception: " + str(value), errorType, traceback)                

def reset_weather(client):
    weathers = [
        # (-1,'defualt'),
        (airsim.WeatherParameter.Rain,"Rain"),
        (airsim.WeatherParameter.Snow,"Snow"),
        # # (airsim.WeatherParameter.MapleLeaf,"MapleLeaf"),
        # # (airsim.WeatherParameter.Roadwetness,"Roadwetness"),
        # (airsim.WeatherParameter.RoadLeaf,"RoadLeaf"),
        # (airsim.WeatherParameter.RoadSnow,"RoadSnow"),
        (airsim.WeatherParameter.Dust,"Dust"),
        (airsim.WeatherParameter.Fog,"Fog"),   
    ]
    # 启用天气效果
    client.simEnableWeather(True)
    weather = choice(weathers)
    if weather[0] in [4,5,6,7]:
        weather_degree = uniform(0 , 0.15)
        client.simSetWeatherParameter(weather[0], uniform(0 , 0.15))
    else:
        weather_degree = uniform(0 , 0.5)
        client.simSetWeatherParameter(weather[0], uniform(0 , 0.5))
    print("current_weathers:%s"%(weather[1]))
    return weather[0],weather[1],weather_degree

# 计算每帧步距
def caculate_step(x_distance_max,y_distance_max,z_distance_max,totalframes):
    x = Symbol('x')
    y = Symbol('y')
    solution=solve([x_distance_max*y_distance_max*24-totalframes*x*y,x_distance_max*y-x*y_distance_max],[x,y])
    x_step = int(abs(solution[0][0]))
    y_step = int(abs(solution[0][1]))
    z_step = z_distance_max/3
    x_circle = int(x_distance_max/abs(solution[0][0]))
    y_circle = int(y_distance_max/abs(solution[0][1]))
    # x_step = round(abs(solution[0][0]))
    # y_step = round(abs(solution[0][1]))
    # z_step = z_distance_max/3
    # x_circle = round(x_distance_max/abs(solution[0][0]))
    # y_circle = round(y_distance_max/abs(solution[0][1]))
    return x_step,y_step,z_step,x_circle,y_circle

def main(record_dir,x_distance_max,y_distance_max,z_distance_max,totalframes):
    # 初始化
    client = init_client()
    x_step,y_step,z_step,x_circle,y_circle = caculate_step(x_distance_max,y_distance_max,z_distance_max,totalframes)
    # 保证步距不小于1.5m
    while True:
        if x_step <=1.5 or y_step<=1.5:
            totalframes = totalframes-500
            x_step,y_step,z_step,x_circle,y_circle = caculate_step(x_distance_max,y_distance_max,z_distance_max,totalframes)
            if x_step >1.5 and y_step>1.5:
                break
        else:
            break
    print(f"循环次数:{x_circle*y_circle*3}")
    # # 随机生成15横轴组线
    # for line_count in range(15):
    #     drone_ned1 =(randint(0,x_distance_max+15), -15, randint(-z_distance_max-3,-2))
    #     drone_ned2 =(randint(0,x_distance_max+15), y_distance_max+15, randint(-z_distance_max-3,-2))
    #     write_line(client,drone_ned1,drone_ned2)
    # # 随机生成15纵轴组线
    # for line_count in range(15):
    #     drone_ned1 =(-15, randint(0,y_distance_max+15), randint(-z_distance_max-3,-2))
    #     drone_ned2 =(x_distance_max+15, randint(0,y_distance_max+15), randint(-z_distance_max-3,-2))
    #     write_line(client,drone_ned1,drone_ned2)
    # 创建data目录
    make_record_dir(record_dir)
    circle_count = 0
    x_single_circle = int(x_circle/2)
    y_single_circle = int(y_circle/2)
    # x_single_circle = round(x_circle/2)
    # y_single_circle = round(y_circle/2)
    for z_count in range(3):
        for x_count in range(-x_single_circle+1,x_single_circle):
            for y_count in range(-y_single_circle+1,y_single_circle):
                if "_weather_random" in record_dir:
                    weather, weather_str, weather_degree = reset_weather(client)
                else:
                    weather,weather_str, weather_degree = -1, "defualt", 1
                weather_record = (weather_str,weather_degree)
                x_step = x_step + uniform(-x_step*0.2,x_step*0.2)
                y_step = y_step + uniform(-y_step*0.2,y_step*0.2)
                # 重置坐标
                x_distance =  x_count*x_step
                y_distance =  y_count*y_step
                z_distance = -z_count*z_step-1.5
                if z_count == 2:
                    z_distance = -z_distance_max-1
                if circle_count % 5 == 0:
                    # 随机生成6组线
                    print("随机生成3组线")
                    for line_count in range(3):
                        # # 横轴组线
                        # drone_ned1 =((x_distance+x_distance_max), y_distance, randint(-z_distance_max,-2))
                        # drone_ned2 =((x_distance-x_distance_max), (y_distance+randint(3*y_step,5*y_step)), randint(-z_distance_max,-2))
                        # 纵轴组线
                        drone_ned1 =((x_distance+uniform(-2*x_step,0)), (y_distance+y_distance_max),uniform(-z_distance_max,-2))
                        drone_ned2 =((x_distance+uniform(0,2*x_step)),(y_distance-y_distance_max), uniform(-z_distance_max,-2))
                        write_line(client,drone_ned1,drone_ned2)
                circle_count += 1
                # drone_ned1 =(x_step+randint(-5,5), y_step+randint(-5,5), z_step+randint(-5,2))
                # drone_ned2 =(x_step+randint(-5,5), y_step+randint(-5,5), z_step+randint(-5,2))
                # write_line(client,drone_ned1,drone_ned2)
                # 八个方向采集
                for i in range(8):
                    yaw_degree = 45*i
                    reset_drone(client,x_distance,y_distance,z_distance,yaw_degree) 
                    n,e,d = get_cur_position(client) # get status information 
                    collision = client.simGetCollisionInfo().has_collided # get status information 
                    print('current position:', '%.2f' % n, '%.2f' % e, '%.2f' % d)
                    print('current collision:', collision)
                    # 开始采集
                    # record_by_Yaw(client, record_dir, z_distance)
                    record_by_Yaw(client, record_dir, z_distance,weather_record)
                #清除天气，避免叠加
                client.simSetWeatherParameter(weather, 0.0)
    # release the drone
    client.reset()
    # client.armDisarm(False)
    # client.enableApiControl(False)

if __name__ == "__main__":
    #每个场景一个目录，不同的场景不能共用一个目录
    mapname = "Sci-fiRoomsandCorridors"
    data_path = "D:\\AirSim\\PythonClient_pano\\airsim_data_pipeline\\output"
    data_dir = os.path.join(data_path, mapname)

    config_dir = "./"
    input_config = load_config(os.path.join(config_dir, 'config.ini'))
    # 地图的x,y,z最大距离
    # x_distance_max = 100
    # y_distance_max = 100
    # z_distance_max = 15
    x_distance_max = int(input_config[mapname][0][1])
    y_distance_max = int(input_config[mapname][1][1])
    z_distance_max = int(input_config[mapname][2][1])
    # 总帧数
    totalframes= 2000
    for i in range(2):
    # for i in range(1):
    # for i in range(1,2):
        if i == 0:
            record_dir = data_dir
            main(record_dir,x_distance_max,y_distance_max,z_distance_max,totalframes)
        else:
            record_dir = data_dir + "_weather_random"
            main(record_dir,x_distance_max,y_distance_max,z_distance_max,totalframes)
    