#using pytorch, numpy, pandas, matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
from torch.optim.lr_scheduler import StepLR  # Import a scheduler (example: StepLR)
import argparse 
import os 

#make a class
def Classifier(input_size, hidden_size, num_classes):
    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            out = torch.relu(self.fc1(x))
            out = self.fc2(out)
            return out
    return Classifier()


def load_and_preprocess_data(file_path, input_start = 1, target_column = 0, test_size=0.2, val_size=0.1, random_state=42):
    """
    Load data from an Excel file, preprocess it, and split into train, validation, and test sets.
    """
    # Load data
    data = pd.read_csv(file_path) 

    # Extract input features and target labels 
    X = data.iloc[input_start:].values #input features
    y = data[target_column].values #target column  

    # Normalize input features
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Split data into train, validation, and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=random_state)
    val_size_adjusted = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state)

    # Convert to PyTorch tensors
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    return train_dataset, val_dataset, test_dataset

# Example usage:
# file_path = '/data/HS_data_for_analysis.csv'
# input_start = ['feature1', 'feature2', 'feature3']  # Replace with actual column names
# target_column = 'target'  # Replace with the actual target column name
# train_dataset, val_dataset, test_dataset = load_and_preprocess_data(file_path, input_start, target_column)

# Create DataLoaders
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

def train_validate_test(model, train_loader, val_loader, test_loader, num_epochs, criterion, optimizer, device, scheduler=None):
    model.to(device)
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Step the scheduler (if provided)
        if scheduler is not None: 
            scheduler.step()

        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Testing loop
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Example usage:
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR by 10x every 10 epochs
# train_validate_test(model, train_loader, val_loader, test_loader, num_epochs=20, criterion=criterion, optimizer=optimizer, device=device, scheduler=scheduler)


def main():
    file_path = '/home/rogue/codeSpace/repos/LinuxCodeBackUp/code/python/thesis/data/HS_data_for_analysis.csv'  # Path to your data file
    # Check if the file exists 
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
    
    # Pass parameters
    parser = argparse.ArgumentParser(description="HS Classification")  
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'], help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default=None, choices=['step', 'cosine'], help='Learning rate scheduler to use')
    parser.add_argument('--input_start', type=int, default=1, help='Row index to start reading input features')
    parser.add_argument('--target_column', type=int, default=0, help='Column index of the target variable')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden layer size')
    args = parser.parse_args()

    # Load data
    train_dataset, val_dataset, test_dataset = load_and_preprocess_data(file_path, args.input_start, args.target_column)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) 
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Define model
    input_size = train_dataset.tensors[0].shape[1]
    num_classes = len(np.unique(train_dataset.tensors[1].numpy())) 
    model = Classifier(input_size, args.hidden_size, num_classes)

    # Define optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif args.optimizer == 'sgd': 
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9)

    # Define scheduler
    scheduler = None
    if args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs) 

    # Define loss function
    criterion = nn.CrossEntropyLoss()  

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    # Train, validate, and test the model
    train_validate_test(model, train_loader, val_loader, test_loader, args.num_epochs, criterion, optimizer, device, scheduler)

if __name__ == "__main__":
    main()