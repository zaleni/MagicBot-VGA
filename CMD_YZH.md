# 新开一个窗口 启动主节点， 一台机器只能启动一个
```bash
roscore
```

## 新开一个窗口启动摄像机节点
```bash
conda activate deploy
source /home/admin1/MagicBot/Deploy/tmp/OrbbecSDK_develop/orbbec_ws/devel/setup.bash 
roslaunch orbbec_camera multi_camera.launch
```


## 新开一个窗口，启动执行动作的节点
### 查看can设备是否激活成功
```bash
cd /home/admin1/MagicBot/Deploy/Piper_ros
bash can_activate.sh
```

### 运行节点
```bash
cd /home/admin1/MagicBot/Deploy/gui_4_2
source /home/admin1/MagicBot/Deploy/Piper_ros/devel/setup.bash
conda activate deploy
roslaunch piper start_single_piper.launch can_port:=can0 auto_enable:=true



## 新开一个窗

cd /home/admin1/deploy
conda activate deploy
bash inference/scripts/pi05_piper_run_yzh.sh



#常用指令
rostopic list

rostopic echo /joint_states_single



rostopic pub -1 /js_cmd sensor_msgs/JointState "{name: ['joint0','joint1','joint2','joint3','joint4','joint5','joint6'], position: [-90346,36605,-46908,831,66802,1428,100000]}"
