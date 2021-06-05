# im-structure


# data
## structurenet

- dataset: http://download.cs.stanford.edu/orion/structurenet/partnethiergeo.zip

- pretrained models: http://download.cs.stanford.edu/orion/structurenet/pretrained_models.zip

## 生成json文件（结构）和 npz（几何） 

- 下载链接: http://download.cs.stanford.edu/orion/partnet_dataset/data_v0.zip
    - [more information](https://www.shapenet.org/download/parts)

- 生成json
    - 处理代码： https://github.com/dclcs/partnet_edges/
    -  先运行 `partnet_edges/detect_all_edges.py`  再运行`prepare_partnetobb_dataset.py`
- 生成npz
    - `partnet_edges/partnet_edges/detect_all_edges.py` 已经生成npz了

## 渲染图片
- 使用 FUTURE-ToolBox ： https://github.com/dclcs/FUTURE3D-ToolBox



