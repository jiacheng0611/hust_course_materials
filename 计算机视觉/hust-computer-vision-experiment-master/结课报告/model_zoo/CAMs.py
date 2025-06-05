import numpy as np
import torch
import torch.nn as nn
import ttach as tta
from numpy import ndarray
from torch.nn.modules import Module
from tqdm import tqdm

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


class CAM(nn.Module):
    """
    CAM 算法基类
    """

    def __init__(self, base_model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        """
        :params base_model: 原始模型
        :params target_layer: 原始模型中的指定层
        """
        super().__init__()
        self.model = base_model
        self.target_layers = target_layer
        self.target = None
        self.device = params["device"]
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layer)
        self.transform = None
        if params["use_aug"]:
            self.transform = tta.Compose([
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ])
        self.name = None

    def get_cam_image(self, inputs: torch.Tensor, activations: torch.Tensor, grads: torch.Tensor) -> np.ndarray:
        """
        获取单层的 CAM 矩阵
        :params inputs: 输入向量
        :params activations: 激活层参数
        :params grads: 梯度
        :params use_eigen: 是否使用主成分降噪
        :returns: CAM 矩阵
        """
        raise NotImplementedError()

    def forward(self, input_tensor: torch.Tensor) -> np.ndarray:
        if params["use_aug"]:
            cams = []
            for transform in self.transform:
                transformed_input_tensor = transform.augment_image(
                    input_tensor)
                params["use_aug"] = False
                cam = self.forward(transformed_input_tensor)
                params["use_aug"] = True
                cam = torch.from_numpy(cam[:, None, :, :])
                cam = transform.deaugment_mask(cam).numpy()
                cams.append(cam[:, 0, :, :])
            cam = np.mean(np.float32(cams), axis=0)
            return cam
        else:
            input_tensor = input_tensor.to(self.device)

            self.outputs = self.activations_and_grads(input_tensor)
            _, predicted = torch.max(self.outputs.data, 1)
            pred = predicted.item()

            if params["flip"]:
                pred = 1 - pred

            self.target = pred

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

        cam = self.get_cam_image(input_tensor, layer_activations, layer_grads)
        cam = np.maximum(cam, 0)
        cam = scale_cam_image(cam, target_size)[:, None, :].squeeze(axis=1)
        return scale_cam_image(cam)

    def __del__(self):
        self.activations_and_grads.release()


class GradCAM(CAM):
    """
    GradCAM 算法
    """

    def __init__(self, base_model: Module, target_layer: Module) -> None:
        """
        :params base_model: 原始模型
        :params target_layer: 原始模型中的指定层
        """
        super().__init__(base_model, target_layer)
        self.name = "GradCAM"

    def get_cam_image(self, inputs: torch.Tensor, activations: torch.Tensor, grads: torch.Tensor) -> np.ndarray:
        """
        获取单层的 CAM 矩阵
        :params inputs: 输入向量
        :params activations: 激活层参数
        :params grads: 梯度
        :returns: CAM 矩阵
        """
        weights = np.mean(grads, axis=(2, 3))
        weighted_activations = weights[:, :, None, None] * activations
        cam = weighted_activations.sum(axis=1)
        if params["use_eigen"]:
            cam = project(weighted_activations)
        return cam


class LayerCAM(CAM):
    """
    LayerCAM 算法
    """

    def __init__(self, base_model: Module, target_layer: Module) -> None:
        super().__init__(base_model, target_layer)
        self.name = "LayerCAM"

    def get_cam_image(self, inputs: torch.Tensor, activations: torch.Tensor, grads: torch.Tensor):
        """
        获取单层的 CAM 矩阵
        :params inputs: 输入向量
        :params activations: 激活层参数
        :params grads: 梯度
        :returns: CAM 矩阵
        """
        spatial_weighted_activations = np.maximum(grads, 0) * activations
        cam = spatial_weighted_activations.sum(axis=1)
        if params["use_eigen"]:
            cam = project(spatial_weighted_activations)
        return cam


class GradCAMpp(CAM):
    """
    GradCAM++ 算法
    """

    def __init__(self, base_model: Module, target_layer: Module) -> None:
        super().__init__(base_model, target_layer)
        self.name = "GradCAM++"

    def get_cam_image(self, inputs: torch.Tensor, activations: torch.Tensor, grads: torch.Tensor):
        """
        获取单层的 CAM 矩阵
        :params inputs: 输入向量
        :params activations: 激活层参数
        :params grads: 梯度
        :returns: CAM 矩阵
        """
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 0.000001
        aij = grads ** 2 / (2 * grads ** 2 +
                            sum_activations[:, :, None, None] * grads ** 2 * grads + eps)
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=(2, 3))

        weighted_activations = weights[:, :, None, None] * activations
        cam = weighted_activations.sum(axis=1)
        if params["use_eigen"]:
            cam = project(weighted_activations)
        return cam


class XGradCAM(CAM):
    """
    XGradCAM 算法
    """

    def __init__(self, base_model: Module, target_layer: Module) -> None:
        super().__init__(base_model, target_layer)
        self.name = "XGradCAM"

    def get_cam_image(self, inputs: torch.Tensor, activations: torch.Tensor, grads: torch.Tensor) -> ndarray:
        """
        获取单层的 CAM 矩阵
        :params inputs: 输入向量
        :params activations: 激活层参数
        :params grads: 梯度
        :returns: CAM 矩阵
        """
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 1e-7
        weights = grads * activations / \
            (sum_activations[:, :, None, None] + eps)
        weights = weights.sum(axis=(2, 3))

        weighted_activations = weights[:, :, None, None] * activations
        cam = weighted_activations.sum(axis=1)
        if params["use_eigen"]:
            cam = project(weighted_activations)
        return cam


class ScoreCAM(CAM):
    """
    ScoreCAM 算法
    """

    def __init__(self, base_model: Module, target_layer: Module) -> None:
        super().__init__(base_model, target_layer)
        self.name = "ScoreCAM"

    def get_cam_image(self, inputs: torch.Tensor, activations: torch.Tensor, grads: torch.Tensor) -> ndarray:
        """
        获取单层的 CAM 矩阵
        :params inputs:
        :params targets:
        :params activations: 激活层参数
        :params grads: 梯度
        :returns: CAM 矩阵
        """
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(
                size=inputs.shape[-2:])
            activation_tensor = torch.from_numpy(activations)
            activation_tensor = activation_tensor.to(self.device)

            upsampled = upsample(activation_tensor)

            maxs = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).min(dim=-1)[0]

            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / (maxs - mins + 1e-8)

            input_tensors = (inputs[:, None, :, :] *
                             upsampled[:, :, None, :, :]).squeeze()

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            scores = []
            for i in tqdm(range(0, input_tensors.size(0), BATCH_SIZE)):
                batch = input_tensors[i: i + BATCH_SIZE, :]
                outputs = [outs.cpu()[self.target].item()
                           for outs in self.model(batch)]
                scores.extend(outputs)

            scores = torch.Tensor(scores)
            scores = scores.view(activations.shape[0], activations.shape[1])
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()

            weighted_activations = weights[:, :, None, None] * activations
            cam = weighted_activations.sum(axis=1)
            if params["use_eigen"]:
                cam = project(weighted_activations)
            return cam


class HiResCAM(CAM):
    """
    HiResCAM 算法
    """

    def __init__(self, base_model: Module, target_layer: Module) -> None:
        super().__init__(base_model, target_layer)
        self.name = "HiResCAM"

    def get_cam_image(self, inputs: torch.Tensor, activations: torch.Tensor, grads: torch.Tensor) -> ndarray:
        """
        获取单层的 CAM 矩阵
        :params inputs:
        :params targets:
        :params activations: 激活层参数
        :params grads: 梯度
        :returns: CAM 矩阵
        """
        elementwise_activations = grads * activations
        return elementwise_activations.sum(axis=1)


if __name__ == "__main__":
    grad_cam = torch.load("./examples/torch_alex.pth")
    print(grad_cam)
    print(grad_cam.features[10])
