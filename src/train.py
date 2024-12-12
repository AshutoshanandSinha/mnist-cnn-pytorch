import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from model import SimpleMNISTNet

# Global lists for tracking metrics
train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        # Get samples
        data, target = data.to(device), target.to(device)

        # Initialize optimizer
        optimizer.zero_grad()

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = F.nll_loss(y_pred, target)
        train_losses.append(loss.item())

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update progress and accuracy
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    accuracy = 100. * correct / len(test_loader.dataset)  # Calculate accuracy
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))  # Use the calculated accuracy

    test_acc.append(accuracy)  # Append accuracy to the list
    return accuracy  # Return the accuracy value

def validate_model(model, device, train_loader):
    # Check the number of parameters
    num_params = model.count_parameters()
    print(f"Number of parameters: {num_params}")

    # Train the model for one epoch to check accuracy
    model.train()
    correct = 0
    total = 0
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    training_accuracy = 100. * correct / total
    print(f"Training accuracy after 1 epoch: {training_accuracy:.2f}%")

    # Validate conditions
    if num_params < 25000 and training_accuracy >= 95.0:
        print("Model validation successful: Meets the criteria.")
    else:
        raise ValueError("Model validation failed: Does not meet the criteria.")

def train_and_test():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model initialization
    network = SimpleMNISTNet().to(device)
    network.print_model_summary()

    # Calculate dataset statistics
    initial_transform = transforms.Compose([transforms.ToTensor()])
    temp_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=initial_transform
    )
    mean, std = calculate_dataset_statistics(temp_dataset)
    print(f"Dataset statistics - Mean: {mean:.4f}, Std: {std:.4f}")

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    transform_train = transforms.Compose([
        transforms.RandomRotation((-5, 5)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

    # Datasets and Loaders
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Validate the model
    validate_model(network, device, train_loader)

    # Optimizer and Scheduler
    optimizer = optim.Adam(
        network.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-5
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        steps_per_epoch=len(train_loader),
        epochs=20,
        pct_start=0.2,
        div_factor=10.0,
        final_div_factor=100.0,
        anneal_strategy='cos'
    )

    # Optional: Plot training metrics
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Progression')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(test_acc, label='Test Accuracy')
    plt.title('Accuracy Progression')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_metrics.png')

def calculate_dataset_statistics(dataset):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=4
    )

    mean = 0.
    std = 0.
    total_images = 0

    for images, _ in tqdm(loader, desc="Calculating dataset statistics"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean.item(), std.item()

if __name__ == "__main__":
    train_and_test()
