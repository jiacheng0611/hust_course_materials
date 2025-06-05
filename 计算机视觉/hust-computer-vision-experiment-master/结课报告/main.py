"""
@Description :   运行可解释性算法
@Author      :   tqychy 
@Time        :   2023/12/20 22:10:42
"""
import json

import torch
from skimage.segmentation import mark_boundaries
from torch.utils.data import DataLoader

from dataset import *
from dataset import PicDataset
from model_zoo.CAMs import *
from model_zoo.LIME import LimeImageExplainer
from utils import *

device = params["device"]
labels = ["cat", "dog"]
cams = {
    "GradCAM": GradCAM,
    "LayerCAM": LayerCAM,
    "GradCAM++": GradCAMpp,
    "ScoreCAM": ScoreCAM,
    "XGradCAM": XGradCAM,
    "HiResCAM": HiResCAM
}


def main_cam(gen_result_image=True, gen_feature_maps=False):
    """
    CAM 算法主函数，用于生成结果热力图和最后一层卷积层的特征图。
    :params gen_result_image: 是否绘制结果热力图
    :params gen_feature_maps: 是否绘制最后一层卷积层的特征图
    """
    set_seed(params["random_seed"])

    # 加载数据
    test_data = PicDataset(params["data_path"])
    test_data_loader = DataLoader(test_data, batch_size=1)
    net = torch.load(params["net_path"]).to(device).eval()
    cam_algorithm = cams[params["cam_type"]](net, net.features).to(device)

    # 准备结果存储位置
    params_path = os.path.join(
        "results", f"name-{cam_algorithm.name}-flip-{params['flip']}-aug-{params['use_aug']}-eigen-{params['use_eigen']}")
    os.makedirs(params_path, exist_ok=True)
    with open(os.path.join(params_path, "params.json"), "w") as p:
        json.dump(params, p)

    # 开始可解释性分析
    for i, inputs in enumerate(test_data_loader):
        inputs = inputs.to(device)
        outputs = net(inputs).to(device)
        _, predicted = torch.max(outputs.data, 1)
        print(f"Test Case {i}: {labels[predicted.item()]}")

        cam = cam_algorithm(inputs).squeeze()  # size: (224, 224) 灰度图
        origin_img = inputs.squeeze().permute(
            1, 2, 0).cpu().numpy()  # size: (224, 224, 3) 原始图片

        if gen_result_image:
            save_result_image(
                origin_img, cam, os.path.join(params_path, f"test{i}.png"))  # 生成热力图
        if gen_feature_maps:
            save_feature_maps(cam_algorithm.feature_maps, os.path.join(
                params_path, f"feature-map{i}.png"))


def main_lime(gen_result_image=True):
    """
    LIME 算法的主函数
    :params gen_result_image: 是否绘制结果热力图
    """
    set_seed(params["random_seed"])

    # 加载数据
    net = torch.load("./examples/torch_alex.pth")
    test_data = PicDataset("./examples/data4")
    test_data_loader = DataLoader(test_data, batch_size=1)

    # 准备结果存储位置
    params_path = os.path.join("results", f"name-LIME-flip-{params['flip']}")
    os.makedirs(params_path, exist_ok=True)
    with open(os.path.join(params_path, "params.json"), "w") as p:
        json.dump(params, p)

    for i, inputs in enumerate(test_data_loader):
        inputs = inputs.squeeze().permute(1, 2, 0).numpy()
        explainer = LimeImageExplainer()
        explanation = explainer.explain_instance(inputs, net, num_samples=1000)
        print(explanation.score)
        if gen_result_image:
            if params["flip"]:
                temp, mask = explanation.get_image_and_mask(
                    explanation.top_labels[1], hide_rest=False, negative_only=False, positive_only=False)
            else:
                temp, mask = explanation.get_image_and_mask(
                    explanation.top_labels[0], hide_rest=False, negative_only=False, positive_only=False)
            plt.imsave(os.path.join(params_path, f"test{i}.png"), mark_boundaries(
                temp, mask))


if __name__ == "__main__":
    # 生成 CAM 热力图
    for cam_type in cams.keys():
        params["cam_type"] = cam_type
        for flip in [False, True]:
            params["flip"] = flip
            for use_aug in [True, False]:
                params["use_aug"] = use_aug
                for use_eigen in [False, True]:
                    if cam_type == "HiResCAM" and use_eigen == True:
                        continue
                    params["use_eigen"] = use_eigen
                    print(
                        f"Algorithm: {cam_type} Flip: {flip} Use Eigen: {use_eigen} Use Aug: {use_aug}")
                    main_cam()

    # 生成 LIME 热力图
    for flip in [False, True]:
        params["flip"] = flip
        print(f"Flip: {flip}")
        main_lime()
