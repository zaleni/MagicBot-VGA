# Enviroment setup
conda create -n liberoplus python==3.9
首先配置libero自己的环境依赖，不要pip install -e .
安装libero-plus
pip install tyro matplotlib mediapy websockets msgpack
pip install mujoco==3.2.3 #可选
mv /root/.libero/config.yaml /root/.libero/config.yaml.bak #关键
如果报找不到Pyyaml等包，回到libero目录 pip install -r requirements.txt