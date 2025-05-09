"""
This file is for training models on the specified dataset.
"""
import os
import glob
import re
import wandb
from torch import optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.models import swin_t, swin_s
from utils.model import *
from utils.utils import set_logger, get_file_name
from utils.curvature_tuning import replace_module
from loguru import logger
from utils.data import get_data_loaders
import argparse
import copy
from sklearn.linear_model import LogisticRegression

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train_epoch(epoch, model, trainloader, optimizer, criterion, device, warmup_scheduler, beta=None):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            partial_loss = running_loss / (batch_idx + 1)
            partial_accuracy = 100. * correct / total
            logger.info(f'Epoch {epoch}, Step {batch_idx}, Loss: {partial_loss:.6f}, Accuracy: {partial_accuracy:.2f}%')

        if epoch <= 1 and warmup_scheduler is not None:
            warmup_scheduler.step()

    # Compute final epoch loss and accuracy
    train_loss = running_loss / len(trainloader)
    train_accuracy = 100. * correct / total

    if beta is None:
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_accuracy,
                   'lr': optimizer.param_groups[0]['lr']})
    else:
        wandb.log({f'epoch_beta{beta:.2f}': epoch, f'train_loss_beta{beta:.2f}': train_loss,
                   f'train_accuracy_beta{beta:.2f}': train_accuracy,
                   f'lr_beta{beta:.2f}': optimizer.param_groups[0]['lr']})

    return train_loss


def test_epoch(epoch, model, testloader, criterion, device, beta=None):
    """
    Test the model for one epoch.
    Specify epoch=-1 to use for testing after training.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(testloader)
    test_accuracy = 100. * correct / total
    if epoch != -1:
        logger.info(f'Epoch {epoch}, Val Loss: {test_loss:.6f}, Val Accuracy: {test_accuracy:.2f}%')

        # Log the test loss and accuracy to wandb
        if beta is None:
            wandb.log({'epoch': epoch, 'val_loss': test_loss, 'val_accuracy': test_accuracy})
        else:
            wandb.log({f'epoch_beta{beta:.2f}': epoch, f'val_loss_beta{beta:.2f}': test_loss,
                       f'val_accuracy_beta{beta:.2f}': test_accuracy})

    else:
        logger.info(f'Loss: {test_loss:.6f}, Accuracy: {test_accuracy:.2f}%')

    return test_loss, test_accuracy


def extract_features_and_labels(feature_extractor, dataloader):
    feature_extractor.eval()
    features_list, labels_list = [], []
    with torch.inference_mode():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            feats = feature_extractor(x)
            feats = torch.flatten(feats, 1)
            features_list.append(feats.cpu())
            labels_list.append(y.cpu())

    return torch.cat(features_list), torch.cat(labels_list)


def linear_probe(model, train_loader, val_loader, beta=None, new_train_batch_size=32, new_val_batch_size=800, deterministic=False):
    """
    Linear probing by extracting features using the frozen backbone (excluding the classifier),
    then training a new linear classifier on those features.
    """
    # Strip the classification head
    feature_extractor = nn.Sequential(*list(model.children())[:-1])

    feature_extractor = feature_extractor.to(device)

    # Extract train/val features
    train_feats, train_labels = extract_features_and_labels(feature_extractor, train_loader)
    val_feats, val_labels = extract_features_and_labels(feature_extractor, val_loader)

    criterion = nn.CrossEntropyLoss()

    if deterministic:
        val_dataset = torch.utils.data.TensorDataset(val_feats, val_labels)
        val_loader_new = torch.utils.data.DataLoader(val_dataset, batch_size=new_val_batch_size, shuffle=False, num_workers=6)
        train_feats, train_labels= train_feats.numpy(), train_labels.numpy()
        logistic_regression = LogisticRegression(max_iter=10000)
        logistic_regression.fit(train_feats, train_labels)
        if hasattr(model, 'fc'):
            model.fc.weight.data = torch.tensor(logistic_regression.coef_, dtype=torch.float32).to(device)
            model.fc.bias.data = torch.tensor(logistic_regression.intercept_, dtype=torch.float32).to(device)
        elif hasattr(model, 'head'):
            model.head.weight.data = torch.tensor(logistic_regression.coef_, dtype=torch.float32).to(device)
            model.head.bias.data = torch.tensor(logistic_regression.intercept_, dtype=torch.float32).to(device)
        else:
            raise RuntimeError('Unknown model architecture')
        _, best_acc = test_epoch(-1, model, val_loader_new, criterion, device)
    else:
        # Create feature datasets
        train_dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
        val_dataset = torch.utils.data.TensorDataset(val_feats, val_labels)
        train_loader_new = torch.utils.data.DataLoader(train_dataset, batch_size=new_train_batch_size, shuffle=True, num_workers=6)
        val_loader_new = torch.utils.data.DataLoader(val_dataset, batch_size=new_val_batch_size, shuffle=False, num_workers=6)

        # Train a linear classifier
        num_features = train_feats.shape[1]
        num_classes = train_labels.max().item() + 1
        classifier = nn.Linear(num_features, num_classes).to(device)
        optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
        warmup_scheduler = WarmUpLR(optimizer, len(train_loader_new))
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

        best_classifier = None
        best_acc = 0.0

        for epoch in range(1, 21):
            train_epoch(epoch, classifier, train_loader_new, optimizer, criterion, device, warmup_scheduler, beta)
            _, val_acc = test_epoch(epoch, classifier, val_loader_new, criterion, device, beta)
            if val_acc > best_acc:
                best_classifier = copy.deepcopy(classifier)
                best_acc = val_acc
                logger.info(f'New best validation accuracy: {val_acc:.2f} at epoch {epoch}')
            scheduler.step()

        # Replace the classifier in the original model with the trained one
        if hasattr(model, 'fc'):
            model.fc = best_classifier
        elif hasattr(model, 'head'):
            model.head = best_classifier
        else:
            raise RuntimeError('Unknown model architecture')

    return model, best_acc


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimizer (e.g., SGD)
        total_iters: total_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Set the learning rate to base_lr * epoch / total_iters"""
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def train(dataset, model_name, batch_size=None, learning_rate=None, num_epochs=None):
    """
    Train the model on the specified dataset.
    :param dataset: dataset to train on, e.g., cifar10/cifar100
    """
    name_to_model = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152,
        'swin_t': swin_t,
        'swin_s': swin_s
    }

    logger.info(f'Training {model_name} on {dataset}...')
    wandb.init(project='curvature-tuning', entity='leyang_hu')

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Hyperparameters
    batch_size = 128 if batch_size is None else batch_size
    learning_rate = 0.1 if learning_rate is None else learning_rate
    num_epochs = num_epochs if num_epochs is not None else 200 if dataset != 'mnist' else 10

    # Get the data loaders
    transform_train, transform_test = None, None
    train_loader, test_loader, _ = get_data_loaders(dataset, train_batch_size=batch_size, transform_train=transform_train, transform_test=transform_test)

    # Initialize the model
    num_classes = 100 if 'cifar100' in dataset or 'imagenet100' in dataset else 10

    if 'swin' not in model_name:
        model = name_to_model[model_name](num_classes=num_classes)
    else:
        model = name_to_model[model_name]()
        model.head = nn.Linear(model.head.in_features, num_classes)
        model = replace_module(model, old_module=nn.GELU, new_module=nn.ReLU)

    model = model.to(device)

    # Create the checkpoint folder
    ckpt_folder = './ckpts'
    os.makedirs(ckpt_folder, exist_ok=True)

    # --- Checkpoint Loading Mechanism ---
    # Look for checkpoint files that follow the pattern: {model_name}_{dataset}_epoch*.pth
    checkpoint_pattern = os.path.join(ckpt_folder, f"{model_name}_{dataset}_epoch*.pth")
    ckpt_files = glob.glob(checkpoint_pattern)
    if ckpt_files:
        # Extract epoch numbers from filenames and find the checkpoint with the highest epoch.
        latest_ckpt = max(
            ckpt_files,
            key=lambda x: int(re.search(f"{model_name}_{dataset}_epoch(\\d+).pth", x).group(1))
        )
        latest_epoch = int(re.search(f"{model_name}_{dataset}_epoch(\\d+).pth", latest_ckpt).group(1))
        logger.info(f"Loading checkpoint {latest_ckpt} from epoch {latest_epoch}...")
        model.load_state_dict(torch.load(latest_ckpt))
        start_epoch = latest_epoch + 1
    else:
        start_epoch = 1
    # -------------------------------------

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # Learning rate scheduler with specific milestones for reduction
    if dataset != 'mnist':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=0.2)

    # Warmup scheduler
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch)

    best_test_loss = float('inf')

    # Train the model starting from the last saved epoch (if any)
    for epoch in range(start_epoch, num_epochs + 1):
        if epoch > 1:
            scheduler.step(epoch)

        train_epoch(epoch, model, train_loader, optimizer, criterion, device, warmup_scheduler)
        test_loss, _ = test_epoch(epoch, model, test_loader, criterion, device)

        # Save every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = os.path.join(ckpt_folder, f'{model_name}_{dataset}_epoch{epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

        # Save the model with the best test loss
        if test_loss < best_test_loss:
            logger.info(f'Found new best model at Epoch {epoch}')
            best_test_loss = test_loss
            best_ckpt_path = os.path.join(ckpt_folder, f'{model_name}_{dataset}_best.pth')
            torch.save(model.state_dict(), best_ckpt_path)

    wandb.finish()
    logger.info(f'Finished training!')
    return model


def get_args():
    parser = argparse.ArgumentParser(description='Training a model on the specified dataset.')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to train on, e.g., cifar10/cifar100/imagenette')
    parser.add_argument('--model', type=str, default='resnet18', help='Model to train, e.g., resnet18')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of epochs')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    f_name = get_file_name(__file__)
    set_logger(name=f'{f_name}_{args.dataset}_{args.model}')
    train(args.dataset, args.model, args.batch_size, args.learning_rate, args.num_epochs)
