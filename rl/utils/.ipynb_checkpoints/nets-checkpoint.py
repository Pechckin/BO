import numpy as np
import torch
import cv2
from torch import nn as nn
import torch.nn.functional as F


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class PreprocessImage(nn.Module):
    def __init__(self, frames=4):
        super(PreprocessImage, self).__init__()
        self.unpack_size = 3136

        self.base = nn.Sequential(
            layer_init(nn.Conv2d(frames, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, state):
        return self.base(state.float())


class ActorCritic(nn.Module):
    def __init__(self, action_dim):
        super(ActorCritic, self).__init__()
        self.preprocess = PreprocessImage()

        self.image_in = nn.Sequential(
            layer_init(nn.Linear(self.preprocess.unpack_size, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(
                256,
                448)),
            nn.ReLU()
        )

        self.extra_layer = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.1),
            nn.ReLU()
        )

        self.actor_out = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(448, action_dim), std=0.01))

        self.critic_out_ext = layer_init(nn.Linear(448, 1), std=1)
        self.critic_out_int = layer_init(nn.Linear(448, 1), std=1)

    def forward(self, x):
        output_image = self.preprocess(x)
        linear_input = self.image_in(output_image)

        actor = self.actor_out(linear_input)
        critic_int = self.critic_out_int(self.extra_layer(linear_input) + linear_input)
        critic_ext = self.critic_out_ext(self.extra_layer(linear_input) + linear_input)
        return actor, critic_int, critic_ext


class AutoEncoder(nn.Module):
    def __init__(self, points_dim: int):
        super(AutoEncoder, self).__init__()

        feature_output = 7 * 7 * 64
        frames = 4

        self.image_out = nn.Sequential(
            layer_init(nn.Conv2d(frames, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.encoder = nn.Sequential(
            layer_init(nn.Linear(feature_output, 512)),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
            layer_init(nn.Linear(512, points_dim)),
            nn.BatchNorm1d(points_dim),
            #nn.Sigmoid(),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            layer_init(nn.Linear(points_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, feature_output))
        )

        self.image_in = nn.Sequential(
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(64, 32, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(32, 4, 8, stride=4)),
            nn.Sigmoid(),

        )

    def get_repr(self, state):
        with torch.no_grad():
            features = self.image_out(state)
            encode = self.encoder(features)
            return encode

    def forward(self, state):
        features = self.image_out(state)
        encode = self.encoder(features)
        decode = self.decoder(encode).view(encode.shape[0], 64, 7, 7)

        state_out = self.image_in(decode)

        return state, state_out
