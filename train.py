import os
import datetime
import argparse
import tqdm
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter
from evnet import EVNet, evnet_params
from utils import set_seed, TinyImageNETDataset

BEST_ACCURACY = 0

def test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int
    ) -> float:

    # Create Loss function
    criterion = nn.CrossEntropyLoss()

    # Set model to eval mode
    model.eval()

    logs_dict = {}
    losses = []

    # List to store predictions and labels
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

            losses.append(criterion(output, y).item())
            predictions.append(prediction.cpu().numpy())
            labels.append(y.cpu().numpy())

        # Concatenate embeddings into a np array of shape: (num_samples, embedding_dim)
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        # Compute loss
        loss = np.mean(losses)
        logs_dict["Validation Loss"] = loss

        # Compute accuracy
        accuracy = (predictions == labels).sum() / labels.shape[0]
        logs_dict["Validation Accuracy"] = accuracy

        # Log to wandb
        writer.add_scalar("Validation Loss", loss, epoch)
        writer.add_scalar("Validation Accuracy", accuracy, epoch)
        writer.flush()

        return loss, accuracy


def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    hyperparameters: dict,
    args: object,
    device: str,
    writer: SummaryWriter,
    checkpoint
    ) -> None:

    global BEST_ACCURACY

    epoch_iterator = range(checkpoint['epoch'] + 1, hyperparameters["epochs"]) if checkpoint else range(hyperparameters["epochs"])
    for epoch in epoch_iterator:
        # Set model to train mode
        model.train()

        # Set training running loss 
        running_loss = 0.

        # Create tqdm progress bar
        with tqdm.tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
            for i, (x, y) in enumerate(train_dataloader):
                # Move batch to device
                x = x.to(device)
                y = y.to(device)

                # Forward pass
                y_hat = model(x)

                # Compute loss
                loss = criterion(y_hat, y)

                # Zero gradients
                optimizer.zero_grad()

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

                # Add batch loss to running loss
                running_loss += loss.item()

        # Log loss to Tensorboard
        writer.add_scalar("Training Loss", running_loss/len(train_dataloader), epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

        # Log gradients to Tensorboard
        for name, param in model.model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(name.split('.')[0] + '/' + '.'.join(name.split('.')[1:]) + '/grad', param.grad, global_step=epoch)
        if hasattr(model, 'voneblock_bottleneck'):
            for name, param in model.voneblock_bottleneck.named_parameters():
                if param is not None:
                    writer.add_histogram('voneblock_bottleneck/' + name, param, global_step=epoch)

        # Evaluate model
        val_loss, accuracy = test(model, test_dataloader, device, writer, epoch)

        # Update learning rate
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Save model, scheduler and optimizer checkpoint
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "val_accuracy": accuracy
            },
            os.path.join(args.out_dir, "checkpoint.pth"),
        )
        if accuracy > BEST_ACCURACY:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": accuracy
                },
                os.path.join(args.out_dir, "checkpoint_best.pth")
            )
            BEST_ACCURACY = accuracy


def init_parser() -> argparse.ArgumentParser:
    # Define args
    parser = argparse.ArgumentParser(
        description="Training model on Tiny ImageNET"
    )
    parser.add_argument(
        "--hyperparameters",
        default="./hyperparameters.yml",
        type=str,
        help="Path to yaml file with hyperparameters",
    )
    parser.add_argument(
        "--data_path",
        default="../tiny-imagenet-200",
        type=str,
        help="Path to the data",
    )
    parser.add_argument(
        "--out_dir",
        default="./output",
        type=str,
        help="Path to the output directory where the model will be saved",
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
        "--seed",
        default=42,
        type=int,
        help="Seed for the random number generators.",
    )
    parser.add_argument(
        "--model_arch",
        default='resnet18',
        choices=['resnet18', 'resnet50', 'vgg16'],
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

    # Define args
    parser = init_parser()
    args = parser.parse_args()
    with open(args.hyperparameters, "r") as f:
        hyperparameters = yaml.safe_load(f)
    if args.model_arch=='vgg16': 
        hyperparameters['lr' ] = .01
        hyperparameters['epochs'] = 100

    # Create output directory
    args.out_dir = os.path.join(args.out_dir, args.model_family)
    if os.path.isdir(args.out_dir):
        if not args.use_checkpoint :
            args.out_dir = os.path.join(
                args.out_dir,
                (
                    args.model_family  + '_' + args.model_arch + "_" +\
                         str(args.seed) + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                )
            )
            os.makedirs(args.out_dir, exist_ok=True)
        else:
            subdir_count = 0
            for subdir in os.listdir(args.out_dir):
                if os.path.isdir(os.path.join(args.out_dir, subdir)) and int(os.path.basename(os.path.normpath(subdir)).split('_')[-2]) == args.seed:
                    subdir_count += 1
                    assert subdir_count == 1, 'More than one possible directory to retrieve checkpoint!'
                    args.out_dir = os.path.join(args.out_dir, subdir)
            if not subdir_count:
                assert False, 'No checkpoint directory found!'

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Set seed
    set_seed(args.seed)
    
    LastTransform = T.Normalize(mean=[.5]*3, std=[.5]*3) if args.model_family in ['base', 'vonenet'] else T.Lambda(lambda x: x)
    transforms_train = T.Compose([
        T.ConvertImageDtype(dtype=torch.float),
        T.Resize((64, 64), antialias=True),
        T.RandomAffine(degrees=30, scale=(1.0, 1.2), translate=(.05, .05), fill=.5),
        T.RandomHorizontalFlip(p=.5),
        LastTransform
    ])
    transforms_test = T.Compose([
        T.ConvertImageDtype(dtype=torch.float),
        T.Resize((64, 64), antialias=True),
        T.Lambda(lambda img: img.expand(3, -1, -1)),
        LastTransform
    ])

    # Set dataset
    data_train = TinyImageNETDataset(args.data_path, True, args.preload, transforms_train)
    data_test = TinyImageNETDataset(args.data_path, False, args.preload, transforms_test)
    dataloader_train = DataLoader(
        data_train,
        batch_size=hyperparameters["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=args.prefetch_factor
        )
    dataloader_test = DataLoader(
        data_test,
        batch_size=hyperparameters["batch_size"],
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor
        )

    # Set model
    model = EVNet(
        **evnet_params[args.model_family], model_arch=args.model_arch,
        image_size=64, visual_degrees=2, num_classes=200, gabor_seed=args.seed
    )
    model.to(device)

    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=hyperparameters["lr"],
        momentum=hyperparameters["momentum"],
        weight_decay=hyperparameters["weight_decay"]
        )
    if hyperparameters['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=hyperparameters["scheduler_factor"],
                patience=hyperparameters["scheduler_patience"]
                )
    elif hyperparameters['scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=hyperparameters["scheduler_step_size"],
                gamma=hyperparameters["scheduler_gamma"]
                )
    
    # Load checkpoint
    checkpoint = None
    if args.use_checkpoint:
        checkpoint = torch.load(os.path.join(args.out_dir, "checkpoint.pth")) 
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    # Create Logger
    writer = SummaryWriter(
        log_dir=args.out_dir,
        comment=(args.model_family)
        )
    print('logdir:', args.out_dir)

    # Train model
    train(
        model, criterion, optimizer, scheduler,
        dataloader_train, dataloader_test, hyperparameters,
        args, device, writer, checkpoint
        )
    
    writer.close()
