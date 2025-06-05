import numpy as np
import torch
import torch.nn as nn

from utils import *


class ActivationsAndGradients:
    """ 
    从指定卷积层提取激活值并注册梯度钩子函数
    """

    def __init__(self, base_model, target_layer):
        """
        :params base_model: 原始模型
        :params target_layer: 原始模型中的指定层
        """
        self.base_model = base_model
        self.gradients = []
        self.activations = []
        self.handles = []
        self.handles.append(
            target_layer.register_forward_hook(self.save_activation))
        self.handles.append(
            target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        self.activations.append(output.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return

        def _store_grad(grad):
            self.gradients = [grad.cpu().detach()] + self.gradients
        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.base_model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM(nn.Module):
    """
    GradCAM 算法
    """

    def __init__(self, base_model: torch.nn.Module, target_layer: torch.nn.Module, flip=False) -> None:
        """
        :params base_model: 原始模型
        :params target_layer: 原始模型中的指定层
        :params flip: 是否绘制另一个标签的热力图 默认 False
        """
        super().__init__()
        self.model = base_model
        self.target_layers = target_layer
        self.device = params["device"]
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layer)
        self.flip = flip
        self.name = "GradCAM"

    def get_cam_image(self, activations: torch.Tensor, grads: torch.Tensor) -> np.ndarray:
        """
        获取单层的 CAM 矩阵
        :params activations: 激活层参数
        :params grads: 梯度
        :returns: CAM 矩阵
        """
        weights = np.mean(grads, axis=(2, 3))
        weighted_activations = weights[:, :, None, None] * activations
        cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self, input_tensor: torch.Tensor) -> np.ndarray:

        input_tensor = input_tensor.to(self.device)

        self.outputs = self.activations_and_grads(input_tensor)
        _, predicted = torch.max(self.outputs.data, 1)
        pred = predicted.item()
        if self.flip:
            pred = 1 - pred  # 绘制另一个标签的热力图

        self.model.zero_grad()
        self.outputs[:, pred].backward()

        return self.compute_cam(input_tensor)

    def compute_cam(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        进行 CAM 的计算
        :input_tensor: 输入张量
        """
        target_size = input_tensor.size(-1), input_tensor.size(-2)

        layer_activations = self.activations_and_grads.activations[0].cpu(
        ).data.numpy()
        layer_grads = self.activations_and_grads.gradients[0].cpu(
        ).data.numpy()
        self.feature_maps = layer_activations

        cam = self.get_cam_image(layer_activations, layer_grads)
        cam = np.maximum(cam, 0)
        cam = scale_cam_image(cam, target_size)[:, None, :].squeeze(axis=1)
        return scale_cam_image(cam)

    def __del__(self):
        self.activations_and_grads.release()


class LayerCAM(nn.Module):
    """
    LayerCAM 算法
    """

    def __init__(self, base_model: torch.nn.Module, target_layer: torch.nn.Module, flip=False) -> None:
        """
        :params base_model: 原始模型
        :params target_layer: 原始模型中的指定层
        :params flip: 是否绘制另一个标签的热力图 默认 False
        """
        super().__init__()
        self.model = base_model
        self.target_layers = target_layer
        self.device = params["device"]
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layer)
        self.flip = flip
        self.name = "LayerCAM"

    def get_cam_image(self, activations: torch.Tensor, grads: torch.Tensor):
        """
        获取单层的 CAM 矩阵
        :params activations: 激活层参数
        :params grads: 梯度
        :returns: CAM 矩阵
        """
        spatial_weighted_activations = np.maximum(grads, 0) * activations
        return spatial_weighted_activations.sum(axis=1)

    def forward(self, input_tensor: torch.Tensor) -> np.ndarray:

        input_tensor = input_tensor.to(self.device)

        self.outputs = self.activations_and_grads(input_tensor)
        _, predicted = torch.max(self.outputs.data, 1)
        pred = predicted.item()
        if self.flip:
            pred = 1 - pred

        self.model.zero_grad()
        self.outputs[:, pred].backward()

        return self.compute_cam(input_tensor)

    def compute_cam(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        进行 CAM 的计算
        :input_tensor: 输入张量
        """
        target_size = input_tensor.size(-1), input_tensor.size(-2)

        layer_activations = self.activations_and_grads.activations[0].cpu(
        ).data.numpy()
        layer_grads = self.activations_and_grads.gradients[0].cpu(
        ).data.numpy()
        self.feature_maps = layer_activations

        cam = self.get_cam_image(layer_activations, layer_grads)
        cam = np.maximum(cam, 0)
        cam = scale_cam_image(cam, target_size)[:, None, :].squeeze(axis=1)
        return scale_cam_image(cam)

    def __del__(self):
        self.activations_and_grads.release()


if __name__ == "__main__":
    grad_cam = torch.load('./实验四模型和测试图片（PyTorch）/torch_alex.pth')
    print(grad_cam)
    print(grad_cam.features[10])
