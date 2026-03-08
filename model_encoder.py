import torch
# import clip
from PIL import Image
import numpy as np
from utils_REN import load_imagemol,load_pretrained_component
import torch.nn as nn
import torch.nn.functional as F
from layers import ResidualLayer


class ImageMol_Regression(nn.Module):
    def __init__(self):
        super(ImageMol_Regression, self).__init__()
        self.img_encoder = load_imagemol()
        self.residual = ResidualLayer(1000, 512)
        # self.relu = nn.ReLU()
        # self.image_linear = nn.Linear(512, 256)
        self.residual2 = ResidualLayer(512, 256)
        self.fc = nn.Linear(512, 1)

    def predict(self, image):
        pass

    def forward(self, image):
        img_embed = self.img_encoder(image)
        # img_embed = img_embed.view(img_embed.size(0), -1)
        # img_feat = self.image_linear(img_embed.float())
        img_embed = self.residual(img_embed)
        # img_embed = self.residual2(img_embed)
        logits = self.fc(img_embed)
        return logits



class ImageMol_tsne(nn.Module):
    def __init__(self):
        super(ImageMol_tsne, self).__init__()
        self.img_encoder = load_imagemol()

    def predict(self, image):
        pass

    def forward(self, image):
        img_embed = self.img_encoder(image)

        return img_embed

    def get_feature(self, image):
        img_embed = self.img_encoder(image)
        return img_embed



class ImageMol(nn.Module):
    def __init__(self, args):
        super(ImageMol, self).__init__()
        self.img_encoder = load_imagemol()

        self.residual = ResidualLayer(1000, 256)
        # self.image_linear = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, 2)

    def predict(self, image):
        pass

    def forward(self, image):
        img_embed = self.img_encoder(image)
        # img_embed = img_embed.view(img_embed.size(0), -1)
        # img_feat = self.image_linear(img_embed.float())

        img_embed = self.residual(img_embed.float())

        logits = self.classifier(img_embed)
        # pred = torch.softmax(logits, dim=1)
        pred = torch.sigmoid(logits)
        return logits

class ImageMolPrompt(nn.Module):
    def __init__(self, args):
        super(ImageMolPrompt, self).__init__()
        self.img_encoder = load_imagemol()
        # self.img_encoder = load_pretrained_component('ckpts/imagemol/CGIP.pth','model_state_dict1')

        self.entity_parameter = nn.Parameter(torch.from_numpy(np.load('ckpts/drkg_rotate/entity_embedding.npy')), requires_grad=False)

        self.residual = ResidualLayer(1024, 256)
        # self.image_linear = nn.Linear(512, 256)
        dim = self.entity_parameter.size(1)
        self.classifier = nn.Linear(256, args.num_classes)
        # self.classifier = nn.Linear(1024, args.num_classes)
        
        self.prompt_linear = nn.Linear(dim, dim // 2)

    def predict(self, image):
        pass

    def _forward_impl(self, x:torch.Tensor):
        x = self.img_encoder.conv1(x)
        x = self.img_encoder.bn1(x)
        x = self.img_encoder.relu(x)
        x = self.img_encoder.maxpool(x)

        x = self.img_encoder.layer1(x)
        x = self.img_encoder.layer2(x)
        x = self.img_encoder.layer3(x)
        x = self.img_encoder.layer4(x)

        x = self.img_encoder.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.img_encoder.fc(x)

        return x

    def forward(self, h, h_id, t, t_id):
        # prompt_h = self.entity_parameter[h_id]
        # prompt_t = self.entity_parameter[t_id]
        # prompt_h = self.prompt_linear(prompt_h)
        # prompt_t = self.prompt_linear(prompt_t)

        h_embed = self._forward_impl(h)
        t_embed = self._forward_impl(t)

        h_embed = h_embed.view(h_embed.size(0), -1)
        t_embed = t_embed.view(t_embed.size(0), -1)
        # img_feat = self.image_linear(img_embed.float())

        embed = torch.cat([h_embed, t_embed], dim=1)
        img_embed = self.residual(embed.float())

        # logits = self.classifier(torch.cat([img_embed, prompt_h, prompt_t], dim=1))
        logits = self.classifier(img_embed)
        # logits = self.classifier(embed)
        # pred = torch.softmax(logits, dim=1)

        return logits

class ImageMolModel(nn.Module):
    def __init__(self, args):
        super(ImageMolModel, self).__init__()
        self.img_encoder = load_imagemol()

        self.residual = ResidualLayer(1024, 256)
        # self.image_linear = nn.Linear(512, 256)
        dim = self.entity_parameter.size(1)
        self.classifier = nn.Linear(256, args.num_classes)
        
        self.prompt_linear = nn.Linear(dim, dim // 2)

    def predict(self, image):
        pass

    def _forward_impl(self, x:torch.Tensor):
        x = self.img_encoder.conv1(x)
        x = self.img_encoder.bn1(x)
        x = self.img_encoder.relu(x)
        x = self.img_encoder.maxpool(x)

        x = self.img_encoder.layer1(x)
        x = self.img_encoder.layer2(x)
        x = self.img_encoder.layer3(x)
        x = self.img_encoder.layer4(x)

        x = self.img_encoder.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.img_encoder.fc(x)

        return x

    def forward(self, h, t):
        # prompt_h = self.entity_parameter[h_id]
        # prompt_t = self.entity_parameter[t_id]
        # prompt_h = self.prompt_linear(prompt_h)
        # prompt_t = self.prompt_linear(prompt_t)

        h_embed = self._forward_impl(h)
        t_embed = self._forward_impl(t)

        h_embed = h_embed.view(h_embed.size(0), -1)
        t_embed = t_embed.view(t_embed.size(0), -1)
        # img_feat = self.image_linear(img_embed.float())

        embed = torch.cat([h_embed, t_embed], dim=1)
        img_embed = self.residual(embed.float())

        # logits = self.classifier(torch.cat([img_embed, prompt_h, prompt_t], dim=1))
        logits = self.classifier(img_embed)
        # pred = torch.softmax(logits, dim=1)

        return logits


