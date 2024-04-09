import argparse
import torch
import torch.nn as nn
from torch.ultis.dataset import Dataset , DataLoader
from torchvision import transforms

from train import *
from .encoder.model import Classifier, VQAModel
from .encoder.encoder import TextEncoder, VisualEncoder
from .import preprocess
from dataset.coco_vqa import VQADataset
from preprocess import train_data, val_data, test_data, classes_to_idx

def arguments():
    parser = argparse.ArgumentParser( add_help=False)
    parser.add_argument('--device', default = 'cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--train_batch_size', default = 256)
    parser.add_argument('--test_batch_size', default =  32)

    parser.add_argument('--img_model_name', dafault = 'resnet50')

    parser.add_argument('--n_classes', default = 2)
    parser.add_argument('--n_layers', default = 1)
    parser.add_argument('--hidden_size', default = 128)
    parser.add_argument('--embedding_dim', default = 64) 
    parser.add_argument('--dropout_prob', default = 0.2)
    parser.add_argument('--lr', default = 1e-3)
    parser.add_argument('--epoches', default =50)
    parser.add_argument('--scheduler_step_size', default = None, help = 'we let it 0.6* number_of_total_epoches')
    parser.add_argument('--gamma', default = 0.1)
    
    return parser

def main(arg):
    arg = arguments()
    if args.scheduler_step_size is None:
        args.scheduler_step_size = args.epochs * 0.6
    device =arg.device
    if arg.n_classes != len(preprocess.classes):
        arg.n_classes = len(preprocess.classes)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    train_dataset = VQADataset(
        train_data,
        classes_to_idx= classes_to_idx,
        transform= transform
    )
    val_dataset = VQADataset(
        val_data,
        classes_to_idx= classes_to_idx,
        transform= transform
    )
    test_dataset = VQADataset(
        test_data,
        classes_to_idx= classes_to_idx,
        transform= transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size= arg.train_batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size= arg.test_batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size= arg.test_batch_size,
        shuffle=False
    )

    train_loader = DataLoader(train_dataset, 
                              batch_size = arg.train_batch_size, 
                              shuffle=True 
                              )
    val_loader = DataLoader( val_dataset, 
                            batch_size = arg.test_batch_size,
                            shuffle=False )
    # test_loader = DataLoader(test_dataset, 
    #                          batch_size = arg.test_batch_size, 
    #                          shuffle=False
    #                          )

    model = VQAModel(
        n_classes= arg.n_classes,
        img_model_name= arg.img_model_name,
        embeddding_dim= arg.embeddding_dim,
        n_layers= arg.n_layers,
        hidden_size= arg.hidden_size,
        dropout_prob= arg.dropout_prob
    ).to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam( model.parameters(), lr=arg.lr )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=arg.scheduler_step_size, gamma= arg.gamma )

    train_loss, val_loss = fit(model,
                                train_loader,
                                val_loader,
                                criterion,
                                optimizer,
                                scheduler,
                                arg.epochs
                            )
    return train_loss, val_loss

if __name__ == "__main__":
    args  = arguments()
    main(args)