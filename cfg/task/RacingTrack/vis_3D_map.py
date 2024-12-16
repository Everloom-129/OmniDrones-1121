import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import os
import sys

def plot_gate(ax, pos, ori, visible=True):
    # 门的尺寸
    width = 1.0  # 门的宽度
    height = 1.0 # 门的高度
    
    # 计算门的四个角点
    yaw = np.radians(ori[2] + 90 )
    dx = width/2 * np.cos(yaw)
    dy = width/2 * np.sin(yaw)
    
    # 门的四个角点
    corners = np.array([
        [pos[0]-dx, pos[1]-dy, pos[2]-height/2],  # 左下
        [pos[0]-dx, pos[1]-dy, pos[2]+height/2],  # 左上
        [pos[0]+dx, pos[1]+dy, pos[2]+height/2],  # 右上
        [pos[0]+dx, pos[1]+dy, pos[2]-height/2],  # 右下
        [pos[0]-dx, pos[1]-dy, pos[2]-height/2],  # 回到左下，闭合
    ])
    
    # 绘制门框
    color = 'blue' if visible else 'red'
    ax.plot(corners[:,0], corners[:,1], corners[:,2], color=color)

def load_gates_from_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    return data['gates']

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vis_2D_map.py <track_name>")
        sys.exit(1)
        
    yaml_file = sys.argv[1]
    track_name = yaml_file.split('.')[0]
    
    # Load gates data from YAML
    gates_data = load_gates_from_yaml(yaml_file)
    
    # 创建图形
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制所有门
    for gate in gates_data:
        pos = gate['pos']
        ori = gate['ori']
        visible = gate.get('visible', True)
        plot_gate(ax, pos, ori, visible)
    
    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Racing Track Gates - {track_name}')
    
    # 设置视角和比例
    ax.view_init(elev=30, azim=45)
    ax.set_box_aspect([1,1,1])
    
    # 设置坐标轴范围
    limit = 10
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([0, limit])
    
    plt.savefig(f'vis/3D_map_{track_name}.jpg')