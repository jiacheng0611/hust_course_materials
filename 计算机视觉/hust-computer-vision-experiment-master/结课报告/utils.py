"""
@Description :   超参数和工具函数
@Author      :   tqychy 
@Time        :   2023/12/20 16:22:30
"""
import inspect
import os
import random
import sys
import types

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.segmentation import felzenszwalb, quickshift, slic

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ['KaiTi']
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (15.2, 8.8)
fontdict = {"fontsize": 15}

params = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 设备
    "random_seed": 1024,  # 随机数种子
    "data_path": "./examples/data4",  # 测试集的路径
    "net_path": "./examples/torch_alex.pth",  # 预训练网络参数的路径

    "cam_type": "LayerCAM",  # 可解释性算法的名称
    "use_eigen": False, # 是否使用 eigen 平滑
    "use_aug": False, # 是否使用 aug 平滑
    "flip": False  # 绘制另一个标签的热力图
}


def set_seed(seed: int):
    """
    设置随机数种子
    :param seed: 随机数种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save(checkpoint: dict, save_path: str, name: str, enable=True):
    """
    保存模型参数
    :param checkpoint: 模型参数信息
    :param save_path: 模型参数保存地址
    :param name: 模型参数名称
    :param enable: 是否启用本函数
    """
    if enable:
        os.makedirs(save_path, exist_ok=True)
        torch.save(checkpoint, os.path.join(save_path, name))

#######################################
#      CAM 工具函数
#######################################


def scale_cam_image(cam, target_size=None):
    """
    对 CAM 图像进行缩放
    :params cam: CAM 矩阵
    :params target_size: 目标大小
    :returns: 结果图像
    """
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result


def save_result_image(img: np.ndarray, mask: np.ndarray, save_path: str) -> None:
    """
    将 CAM 热力图和原图像叠加，生成结果图像
    :param img: 原始图像。
    :param mask: cam热力图。
    :param save_path: 结果图像的存储路径
    """
    _, ax = plt.subplots()
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = ax.figure.colorbar(ax.imshow(mask, cmap='jet', alpha=0.5))
    cbar.ax.set_ylabel('Heatmap Intensity', rotation=90)
    plt.savefig(save_path)
    plt.clf()


def save_feature_maps(feature_maps: torch.Tensor, fig_name: str):
    """
    绘制指定卷积层的特征图
    :params feature_maps: 卷积层的特征图的张量
    :params fig_name: 保存的图片的路径
    """
    feature_maps = feature_maps.squeeze()

    fig = plt.figure()
    for i in range(feature_maps.shape[0]):
        feature_map = feature_maps[i, :, :].squeeze()
        ax = fig.add_subplot(16, 16, i+1)
        cbar = ax.imshow(feature_map, alpha=0.8)
        ax.axis("off")

    cax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(cbar, cax=cax)

    plt.savefig(fig_name, bbox_inches='tight')
    plt.clf()


def project(activation_batch):
    """
    eigen-smooth 的 2D 投影
    :params activation_batch: 激活值加权后的张量 
    """
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []
    for activations in activation_batch:
        reshaped_activations = (activations).reshape(
            activations.shape[0], -1).transpose()

        reshaped_activations = reshaped_activations - \
            reshaped_activations.mean(axis=0)
        _, _, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
        projection = reshaped_activations @ VT[0, :]
        projection = projection.reshape(activations.shape[1:])
        projections.append(projection)
    return np.float32(projections)


#######################################
#      LIME 工具函数
#      来自 https://github.com/marcotcr/lime
#######################################


def has_arg(fn, arg_name):
    """Checks if a callable accepts a given keyword argument.

    Args:
        fn: callable to inspect
        arg_name: string, keyword argument name to check

    Returns:
        bool, whether `fn` accepts a `arg_name` keyword argument.
    """
    if sys.version_info < (3,):
        if isinstance(fn, types.FunctionType) or isinstance(fn, types.MethodType):
            arg_spec = inspect.getargspec(fn)
        else:
            try:
                arg_spec = inspect.getargspec(fn.__call__)
            except AttributeError:
                return False
        return (arg_name in arg_spec.args)
    elif sys.version_info < (3, 6):
        arg_spec = inspect.getfullargspec(fn)
        return (arg_name in arg_spec.args or
                arg_name in arg_spec.kwonlyargs)
    else:
        try:
            signature = inspect.signature(fn)
        except ValueError:
            # handling Cython
            signature = inspect.signature(fn.__call__)
        parameter = signature.parameters.get(arg_name)
        if parameter is None:
            return False
        return (parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                   inspect.Parameter.KEYWORD_ONLY))


class BaseWrapper(object):
    """Base class for LIME Scikit-Image wrapper


    Args:
        target_fn: callable function or class instance
        target_params: dict, parameters to pass to the target_fn


    'target_params' takes parameters required to instanciate the
        desired Scikit-Image class/model
    """

    def __init__(self, target_fn=None, **target_params):
        self.target_fn = target_fn
        self.target_params = target_params

    def _check_params(self, parameters):
        """Checks for mistakes in 'parameters'

        Args :
            parameters: dict, parameters to be checked

        Raises :
            ValueError: if any parameter is not a valid argument for the target function
                or the target function is not defined
            TypeError: if argument parameters is not iterable
         """
        a_valid_fn = []
        if self.target_fn is None:
            if callable(self):
                a_valid_fn.append(self.__call__)
            else:
                raise TypeError('invalid argument: tested object is not callable,\
                 please provide a valid target_fn')
        elif isinstance(self.target_fn, types.FunctionType) \
                or isinstance(self.target_fn, types.MethodType):
            a_valid_fn.append(self.target_fn)
        else:
            a_valid_fn.append(self.target_fn.__call__)

        if not isinstance(parameters, str):
            for p in parameters:
                for fn in a_valid_fn:
                    if has_arg(fn, p):
                        pass
                    else:
                        raise ValueError(
                            '{} is not a valid parameter'.format(p))
        else:
            raise TypeError('invalid argument: list or dictionnary expected')

    def set_params(self, **params):
        """Sets the parameters of this estimator.
        Args:
            **params: Dictionary of parameter names mapped to their values.

        Raises :
            ValueError: if any parameter is not a valid argument
                for the target function
        """
        self._check_params(params)
        self.target_params = params

    def filter_params(self, fn, override=None):
        """Filters `target_params` and return those in `fn`'s arguments.
        Args:
            fn : arbitrary function
            override: dict, values to override target_params
        Returns:
            result : dict, dictionary containing variables
            in both target_params and fn's arguments.
        """
        override = override or {}
        result = {}
        for name, value in self.target_params.items():
            if has_arg(fn, name):
                result.update({name: value})
        result.update(override)
        return result


class SegmentationAlgorithm(BaseWrapper):
    """ Define the image segmentation function based on Scikit-Image
            implementation and a set of provided parameters

        Args:
            algo_type: string, segmentation algorithm among the following:
                'quickshift', 'slic', 'felzenszwalb'
            target_params: dict, algorithm parameters (valid model paramters
                as define in Scikit-Image documentation)
    """

    def __init__(self, algo_type, **target_params):
        self.algo_type = algo_type
        if (self.algo_type == 'quickshift'):
            BaseWrapper.__init__(self, quickshift, **target_params)
            kwargs = self.filter_params(quickshift)
            self.set_params(**kwargs)
        elif (self.algo_type == 'felzenszwalb'):
            BaseWrapper.__init__(self, felzenszwalb, **target_params)
            kwargs = self.filter_params(felzenszwalb)
            self.set_params(**kwargs)
        elif (self.algo_type == 'slic'):
            BaseWrapper.__init__(self, slic, **target_params)
            kwargs = self.filter_params(slic)
            self.set_params(**kwargs)

    def __call__(self, *args):
        return self.target_fn(args[0], **self.target_params)
