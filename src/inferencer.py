import json
from os.path import join
from typing import Any, Callable, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.models as models
from numpy import ndarray
from pandas import DataFrame
from PIL import Image
from PIL.Image import Image as ImageType
from torch import FloatTensor, nn
from torch.nn import Module

from image_handler import ImageHandler


class Inferencer:
    df_coords_columns = ['true_x', 'true_y', 'pred_x', 'pred_y']

    def __init__(self, config: Dict[str, Dict[str, Any]], model_name: str) -> None:
        self.cfg = config
        self.images: List[ImageHandler] = []
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.backbone, self.weight, self.preprocess_fn = self._init_backbone(model_name)

    def add_data_item(self, path_to_img: str, path_to_json: str) -> None:
        with open(path_to_json, 'r') as f:
            coords = json.load(f)[0]
        img = Image.open(path_to_img).convert('RGB')
        self.images.append(ImageHandler(img, coords['x'], coords['y']))

    def predict(self) -> None:
        # init
        self.coords = np.zeros((len(self.images), 4), dtype=np.float32)
        batch_size = int(self.cfg['inference']['batch_size'])
        num_batches = int(np.around(np.ceil(len(self.images) / batch_size), 0))
        ref_X_item = self.preprocess_fn(self.images[0].img).float()  # 4D-tensor: NCHW
        # loop over batches
        for b in range(num_batches):
            # init
            batch = self.images[b * batch_size: (b + 1) * batch_size]
            X = torch.zeros((len(batch), *ref_X_item.shape[1:]), dtype=torch.float)
            shapes = np.zeros((len(batch), 2), dtype=int)
            true_x = np.zeros(len(batch), dtype=np.float32)
            true_y = np.zeros_like(true_x)
            # compose arrays
            for i in range(len(batch)):
                X[i] = self.preprocess_fn(batch[i].img).float()
                shapes[i] = np.asarray(batch[i].img).shape
                true_x[i] = batch[i].true_x
                true_y[i] = batch[i].true_y
            # forward
            conv_feature_maps = self.backbone(X.to(self.device)).cpu().detach().numpy()
            act_maps = self._compute_activation_map(conv_feature_maps, self.weight, shapes)
            flattened_act_maps = act_maps.reshape(act_maps.shape[0], -1)
            flattened_argmins = np.argmin(flattened_act_maps, axis=1)
            pred_x = flattened_argmins % act_maps.shape[1] / act_maps.shape[2]
            pred_y = flattened_argmins // act_maps.shape[1] / act_maps.shape[1]
            # store results
            self.coords[b * batch_size: (b + 1) * batch_size, 0] = true_x
            self.coords[b * batch_size: (b + 1) * batch_size, 1] = true_y
            self.coords[b * batch_size: (b + 1) * batch_size, 2] = pred_x
            self.coords[b * batch_size: (b + 1) * batch_size, 3] = pred_y

    def get_coords(self) -> DataFrame:
        """
        Returns a dataframe with the following columns:
            true_x, true_y, pred_x, pred_y
        """
        return DataFrame(data=self.coords, columns=self.df_coords_columns)

    def _init_backbone(
        self, model_name: str
    ) -> Tuple[Module, ndarray, Callable[[ImageType], FloatTensor]]:
        """
        Return:
            backbone, weight, preprocess
        """
        if model_name == 'inception_v3':
            model = models.inception_v3(aux_logits=False)
            model_weights = torch.load(join(self.cfg['paths']['models'], 'inception_v3_google-0cc3c7bd.pth'))
            model.load_state_dict(model_weights.get_state_dict(progress=True), strict=False)
            backbone = nn.Sequential(*list(model.children())[:-3])
            weight = np.mean(np.squeeze(list(backbone[17].children())[-1].conv.weight.data.numpy()), axis=0)
        elif model_name == 'resnet101':
            model_weights = torch.load(join(self.cfg['paths']['models'], 'resnet101-cd907fc2.pth'))
            model = models.resnet101(weights=model_weights)
            backbone = nn.Sequential(*list(model.children())[:-2])
            weight = np.squeeze(list(backbone.parameters())[-1].data.numpy())
        elif model_name == 'vgg19':
            model_weights = torch.load(join(self.cfg['paths']['models'], 'vgg19-dcbb9e9d.pth'))
            model = models.vgg19(weights=model_weights)
            backbone = nn.Sequential(*list(model.children())[:-1])
            weight = np.mean(np.squeeze(list(backbone.parameters())[-2].data.numpy()), axis=(0, 2, 3))
        else:
            raise ValueError('Unknown model name')
        backbone.eval()
        preprocess_fn = model_weights.transforms()
        return backbone.to(self.device), weight, preprocess_fn

    def _compute_activation_map(
        self, conv_feature_maps: ndarray, weight: ndarray, shapes: ndarray
    ) -> ndarray:
        "Computes and returns Class Activation Map"
        n, c, h, w = conv_feature_maps.shape
        reshaped_conv_feature_map = conv_feature_maps.reshape((n, c, h * w))
        cams = np.zeros((n, h * w), dtype=np.float32)
        for i in range(len(cams)):
            cams[i] = np.matmul(weight, reshaped_conv_feature_map[i])
        cams = cams.reshape(n, h, w)
        cams = cams - np.amin(cams, axis=(1, 2), keepdims=True)
        cams = cams / np.amax(cams, axis=(1, 2), keepdims=True)
        cams = np.uint8(255 * cams)
        for i in range(len(cams)):
            cams[i] = cv2.resize(cams[i], tuple(shapes[i]))
        return cams
