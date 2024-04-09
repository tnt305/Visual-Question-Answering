from PIL import Image
import os
import pandas as pd
import numpy as np
import torch
from torch.ultis.dataset import Dataset , DataLoader
from dataset.preprocess import *


class VQADataset(Dataset):
    def __init__(
        self,
        data,
        classes_to_idx,
        img_feature_extractor,
        text_tokenizer,
        device,
        root_dir='./val2014-resised/'
    ):
        self.data = data
        self.root_dir = root_dir
        self.classes_to_idx = classes_to_idx
        self.img_feature_extractor = img_feature_extractor
        self.text_tokenizer = text_tokenizer
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.data[index]['image_path'])
        img = Image.open(img_path).convert('RGB')
        
        if self.img_feature_extractor:
            img = self.img_feature_extractor(images=img, return_tensors="pt")
            img = {k: v.to(self.device).squeeze(0) for k, v in img.items()}
            
        question = self.data[index]['question']
        if self.text_tokenizer:
            question = self.text_tokenizer(
                                            question, 
                                            padding="max_length", 
                                            max_length=20, 
                                            truncation=True,
                                            return_tensors="pt"
                                         )
            question = {k: v.to(self.device).squeeze(0) for k, v in question.items()}


        label = self.data[index]['answer']
        label = torch.tensor(
            classes_to_idx[label],
            dtype=torch.long
        ).to(device)
            
        sample = {
            'image': img,
            'question': question,
            'label': label
        }
            
        return sample