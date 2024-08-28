import os
import argparse
import tqdm
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from evnet import EVNet, evnet_params
from utils import set_seed, TinyImageNET_CDataset


def test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
    ) -> float:

    # Set model to eval mode
    model.eval()

    # ListS to store predictions and labels
    predictions = []
    labels = []

    # Disable gradient computation during evaluation
    with torch.no_grad():
        for x, y in dataloader:
            # Move batch to device
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            output = model(x)
            prediction = torch.argmax(output, dim=1)

            predictions.append(prediction.cpu().numpy())
            labels.append(y.cpu().numpy())

        # Concatenate embeddings into a np array of shape: (num_samples, embedding_dim)
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        # Compute accuracy
        accuracy = (predictions == labels).sum() / labels.shape[0]

        return accuracy

def init_parser() -> argparse.ArgumentParser:
    # Define args
    parser = argparse.ArgumentParser(
        description="Evaluating model on Tiny ImageNET-C"
    )
    parser.add_argument(
        "--data_path",
        default="../tiny-imagenet-c",
        type=str,
        help="Path to the data",
    )
    parser.add_argument(
        "--use_checkpoint",
        action="store_true",
        help="Whether to load the last checkpoint",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Whether to preload the data in memory",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--prefetch_factor",
        default=2,
        type=int,
        help="Number of batches loaded in advance by each worker.",
    )
    parser.add_argument(
        "--model_arch",
        default='resnet18',
        choices=['resnet18', 'vgg16'],
        type=str,
        help="EVNet backend model architecture",
    )
    parser.add_argument(
        "--model_family",
        default="base",
        choices=['base', 'retinanet', 'vonenet', 'evnet'],
        type=str,
        help="Model family",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="Device to use when training the model",
    )
    return parser
    

if __name__ == "__main__":

    parser = init_parser()
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Set transforms
    LastTransform = T.Normalize(mean=[.5]*3, std=[.5]*3) if args.model_family in ['base', 'vonenet'] else T.Lambda(lambda x: x)  
    transforms = T.Compose([
        T.ConvertImageDtype(dtype=torch.float),
        T.Resize((64, 64), antialias=True),
        T.Lambda(lambda img: img.expand(3, -1, -1)),
        LastTransform
    ])

    # Set model
    directory = os.path.join('./output', args.model_family)
    subdirs = []
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path) and args.retinablock in subdir:
            subdirs.append((
                int(os.path.basename(os.path.normpath(subdir)).split('_')[-2]),  # seed from folder name
                os.path.join(directory, subdir, 'checkpoint_best.pth')  # checkpoint path
                ))

    accuracy_data = {
        'Severity': [],
        'Corruption' : [],
        'Accuracy': [],
        'Seed': [],
        'RetinaBlock': [],
        'Backend': [],
        'Name': []
    }

    for seed, checkpoint_path in subdirs:

        # Set model
        model = EVNet(
            **evnet_params[args.model_family],
            model_arch=args.model_arch, image_size=64,
            visual_degrees=2, num_classes=200
        )
        model.to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        corruptions = [
                folder 
                for folder in os.listdir(args.data_path)
                if os.path.isdir(os.path.join(args.data_path, folder))
                ]
        
        # Update clean accuracy data
        accuracy_data['Severity'].append(0)
        accuracy_data['Corruption'].append('clean')
        accuracy_data['Accuracy'].append(checkpoint["val_accuracy"])
        accuracy_data['Seed'].append(seed)
        accuracy_data['RetinaBlock'].append(args.retinablock)
        accuracy_data['Backend'].append(args.model_arch)
        accuracy_data['Name'].append(args.experiment_name)

        with tqdm.tqdm(total=len(corruptions)*5, desc=f'Seed {seed}') as pbar:
            for corruption in corruptions:
                for severity in range(1, 6):

                    # Set dataset
                    dataset = TinyImageNET_CDataset(args.data_path, corruption, severity, args.preload, transforms)
                    dataloader = DataLoader(
                        dataset,
                        batch_size=128,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        prefetch_factor=args.prefetch_factor
                        )
            
                    accuracy = test(model, dataloader, device)

                    # Update accuracy data
                    accuracy_data['Severity'].append(severity)
                    accuracy_data['Corruption'].append(corruption)
                    accuracy_data['Accuracy'].append(accuracy)
                    accuracy_data['Seed'].append(seed)
                    accuracy_data['RetinaBlock'].append(args.retinablock)
                    accuracy_data['Backend'].append(args.model_arch)
                    accuracy_data['Name'].append(args.experiment_name)
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix(corruption=corruption, severity=severity)

    # Save dataframe to CSV
    pd.DataFrame(accuracy_data).to_csv(f'{args.experiment_name}.csv')
