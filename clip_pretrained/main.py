import argparse
import open_clip
import torch
import torch.nn as nn
from torch.ultis.dataset import Dataset , DataLoader

from train import *
from .encoder.model import Classifier, VQAModel
from .encoder.encoder import TextEncoder, VisualEncoder

from dataset.coco_vqa import VQADataset
from preprocess import train_data, val_data, classes_to_idx, classes

def arguments():
    parser = argparse.ArgumentParser( add_help=False)
    parser.add_argument('--device', default = 'cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--train_batch_size', default = 256)
    parser.add_argument('--test_batch_size', default =  32)

    parser.add_argument('--base_model', default = "'ViT-B-32'")
    parser.add_argument('--pretrained_clip', default = "laion2B-s34B-b79K")

    parser.add_argument('--n_classes', default = 2)
    parser.add_argument('--hidden_size', dafault = 1024)
    parser.add_argument('--n_layers', default = 1)
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
    if arg.n_classes != len(classes):
        arg.n_classes = len(classes)
        
    model_clip, _,img_feature_extractor = open_clip.create_model_and_transforms(arg.base_model, pretrained = arg.pretrained_clip)
    text_tokenizer = open_clip.get_tokenizer(arg.base_model)

    train_dataset = VQADataset(
        train_data,
        classes_to_idx=classes_to_idx,
        img_feature_extractor=img_feature_extractor,
        text_tokenizer= text_tokenizer,
        device=device
    )
    val_dataset = VQADataset(
        val_data,
        classes_to_idx= classes_to_idx,
        img_feature_extractor=img_feature_extractor,
        text_tokenizer= text_tokenizer,
        device=device
    )
#     test_dataset = VQADataset(
#         test_data,
#         classes_to_idx=classes_to_idx,
#         img_feature_extractor=img_feature_extractor,
#         text_tokenizer=text_tokenizer,
#         device=device
# )

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


    text_encoder = TextEncoder(model_clip).to(device)
    visual_encoder = VisualEncoder(model_clip).to(device)
    classifier = Classifier().to(device)

    model = VQAModel(
        visual_encoder=visual_encoder,
        text_encoder = text_encoder,
        classifier=classifier
    ).to(device)

    model.freeze()

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