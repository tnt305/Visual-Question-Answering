import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
from transformers import AutoTokenizer, RobertaModel

'''
The code stays the same as in the original code in 
    https://github.com/LAION-AI/CLIP_benchmark/tree/main

'''

class TextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super(TextEncoder, self).__init__()
        self.text_encoder = text_encoder

    def forward(self, inputs):
        outputs = self.text_encoder.encode_text(inputs)
        text_features = outputs / outputs.norm(dim=-1, keepdim=True)
        return text_features

class VisualEncoder(nn.Module):
    def __init__(self, vision_extractor):
        super(VisualEncoder, self).__init__()
        self.vision_extractor = vision_extractor

    def forward(self, inputs):
        outputs = self.vision_extractor.encode_image(inputs)
        img_features = outputs / outputs.norm(dim=-1, keepdim=True)
        return img_features