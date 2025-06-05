"""
@Description :   运行可解释性算法
@Author      :   tqychy 
@Time        :   2023/12/20 22:10:42
"""
from torch.utils.data import DataLoader

from dataset import PicDataset
from net import *
from utils import *

device = params["device"]
labels = ["cat", "dog"]
cams = {
    "GradCAM": GradCAM,
    "LayerCAM": LayerCAM
}


def main(gen_result_image=True, gen_feature_maps=True):
    """
    CAM 算法主函数，用于生成结果热力图和最后一层卷积层的特征图。
    :params gen_result_image: 是否绘制结果热力图
    :params gen_feature_maps: 是否绘制最后一层卷积层的特征图
    """
    set_seed(params["random_seed"])

    test_data = PicDataset(params["data_path"])
    test_data_loader = DataLoader(test_data, batch_size=1)
    net = torch.load(params["net_path"]).to(device).eval()
    print(net)
    cam_algorithm = cams[params["cam_type"]](
        net, net.features, params["flip"]).to(device)

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
                origin_img, cam, f"./images/{cam_algorithm.name}-test{i}-other.png" if params["flip"] else f"./images/{cam_algorithm.name}-test{i}.png")  # 生成热力图
        if gen_feature_maps:
            save_feature_maps(cam_algorithm.feature_maps,
                              f"./images/feature-map{i}.png")


if __name__ == "__main__":
    # 生成 CAM 热力图
    params["cam_type"] = "GradCAM"
    main()
    params["cam_type"] = "LayerCAM"
    main()

    params["flip"] = True
    params["cam_type"] = "GradCAM"
    main()
    params["cam_type"] = "LayerCAM"
    main()
