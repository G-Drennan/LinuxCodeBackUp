#!/usr/bin/env python
# coding: utf-8

"""
Visualization Utilities for Traffic Sign Classification
-------------------------------------------------
Utilities for visualizing datasets, training results, and model predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report


def visualize_class_examples(X, y, class_names, n_samples=5, figsize=(15, 3 * 5)):
    """
    Visualize example images for each class

    Args:
        X (numpy.ndarray): Image data
        y (numpy.ndarray): Labels
        class_names (dict): Dictionary mapping class indices to names
        n_samples (int): Number of samples to display per class
        figsize (tuple): Figure size
    """
    n_classes = len(set(y))

    if n_classes <= 10:  # Display all classes when number is moderate
        fig, axes = plt.subplots(n_classes, n_samples, figsize=figsize)
        fig.subplots_adjust(hspace=0.3, wspace=0.2)

        for i in range(n_classes):
            # Get indices for current class
            indices = np.where(y == i)[0]

            # Skip if too few samples
            if len(indices) < n_samples:
                continue

            # Randomly select n_samples to display
            selected = np.random.choice(indices, n_samples, replace=False)

            for j, idx in enumerate(selected):
                image = X[idx]
                # Ensure image is 3D, if grayscale, copy to 3 channels
                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=2)

                ax = axes[i, j] if n_classes > 1 else axes[j]
                ax.imshow(image)
                ax.set_title(f"{class_names[i]}")
                ax.axis("off")

        plt.tight_layout()
        plt.show()
    else:
        # If too many classes, only show first 10
        print("Too many classes, showing examples for first 10 classes only")
        visualize_class_examples(X, y, class_names, n_samples, figsize)


def visualize_image_intensity(images):
    """
    Visualize image intensity distribution

    Args:
        images (numpy.ndarray): Image data
    """
    if len(images) == 0:
        print("No image data available for analysis")
        return

    # Calculate average intensity and standard deviation for each image
    intensities = []
    std_devs = []

    for img in images:
        if len(img.shape) == 3:  # Color image
            # Convert to grayscale to calculate intensity
            if img.shape[2] == 3:
                # Color image to grayscale
                gray = (
                    0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
                )
            else:
                gray = img[:, :, 0]  # Assume single channel
        else:
            gray = img

        intensities.append(np.mean(gray))
        std_devs.append(np.std(gray))

    plt.figure(figsize=(12, 5))

    # Average intensity distribution
    plt.subplot(1, 2, 1)
    plt.hist(intensities, bins=50)
    plt.title("Image Average Intensity Distribution")
    plt.xlabel("Average Pixel Intensity")
    plt.ylabel("Number of Images")

    # Standard deviation distribution
    plt.subplot(1, 2, 2)
    plt.hist(std_devs, bins=50)
    plt.title("Image Standard Deviation Distribution")
    plt.xlabel("Pixel Standard Deviation")
    plt.ylabel("Number of Images")

    plt.tight_layout()
    plt.show()


def visualize_before_after_preprocessing(images, n_samples=5, preprocessing_func=None):
    """
    Visualize images before and after preprocessing

    Args:
        images (numpy.ndarray): Original image data
        n_samples (int): Number of samples to display
        preprocessing_func (callable): Preprocessing function, accepts an image and returns processed image
    """
    if len(images) == 0:
        print("No image data available for analysis")
        return

    # Randomly select n_samples
    indices = np.random.choice(len(images), min(n_samples, len(images)), replace=False)

    fig, axes = plt.subplots(2, n_samples, figsize=(15, 6))
    fig.subplots_adjust(hspace=0.3, wspace=0.2)

    for i, idx in enumerate(indices):
        original = images[idx]

        # Ensure image is 3D, if grayscale, copy to 3 channels
        if len(original.shape) == 2:
            original = np.stack([original] * 3, axis=2)

        # Display original image
        axes[0, i].imshow(original)
        axes[0, i].set_title("Original Image")
        axes[0, i].axis("off")

        # Preprocess and display
        if preprocessing_func:
            processed = preprocessing_func(original)

            # If processed image is PyTorch tensor, convert back to numpy array
            if isinstance(processed, torch.Tensor):
                processed = processed.permute(
                    1, 2, 0
                ).numpy()  # From [C,H,W] to [H,W,C]
                # Denormalize (assuming ImageNet mean and std)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                processed = std * processed + mean
                processed = np.clip(processed, 0, 1)

            axes[1, i].imshow(processed)
            axes[1, i].set_title("Preprocessed")
            axes[1, i].axis("off")
        else:
            axes[1, i].text(
                0.5,
                0.5,
                "No preprocessing function provided",
                ha="center",
                va="center",
                transform=axes[1, i].transAxes,
            )
            axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_batch(
    loader, class_names, n_samples=8, figsize=(12, 6), mean=None, std=None
):
    """
    Visualize a batch of images

    Args:
        loader (DataLoader): Data loader
        class_names (dict): Dictionary mapping class indices to names
        n_samples (int): Number of samples to display
        figsize (tuple): Figure size
        mean (list): Mean values for denormalization
        std (list): Standard deviation values for denormalization
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]  # ImageNet mean
    if std is None:
        std = [0.229, 0.224, 0.225]  # ImageNet std

    # Get a batch
    try:
        data_iter = iter(loader)
        batch = next(data_iter)
    except StopIteration:
        print("Data loader is empty")
        return

    # Check if batch includes original images
    include_original = len(batch) > 2

    if include_original:
        images, orig_images, labels = batch
    else:
        images, labels = batch

    # Limit number of samples
    n_samples = min(n_samples, images.size(0))

    if include_original:
        # Display original and preprocessed images
        fig, axes = plt.subplots(2, n_samples, figsize=figsize)

        for i in range(n_samples):
            # Display original image
            orig_img = orig_images[i].permute(1, 2, 0).numpy()
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f"Original: {class_names[labels[i].item()]}")
            axes[0, i].axis("off")

            # Display preprocessed image
            img = images[i].permute(1, 2, 0).numpy()
            # Denormalize
            mean_np = np.array(mean)
            std_np = np.array(std)
            img = std_np * img + mean_np
            img = np.clip(img, 0, 1)

            axes[1, i].imshow(img)
            axes[1, i].set_title("Preprocessed")
            axes[1, i].axis("off")
    else:
        # Only display preprocessed images
        fig, axes = plt.subplots(1, n_samples, figsize=figsize)

        for i in range(n_samples):
            img = images[i].permute(1, 2, 0).numpy()
            # Denormalize
            mean_np = np.array(mean)
            std_np = np.array(std)
            img = std_np * img + mean_np
            img = np.clip(img, 0, 1)

            axes[i].imshow(img)
            axes[i].set_title(f"{class_names[labels[i].item()]}")
            axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_augmentations(image, n_augmentations=5, figsize=(15, 3)):
    """
    Visualize data augmentation effects

    Args:
        image (numpy.ndarray): Original image
        n_augmentations (int): Number of augmented images to generate
        figsize (tuple): Figure size
    """
    # Ensure image is 3D, if grayscale, copy to 3 channels
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=2)

    # Convert to PIL image
    image = Image.fromarray(image.astype(np.uint8))

    # Define augmentation transformations
    augmentations = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.RandomResizedCrop(size=image.size, scale=(0.8, 1.0)),
        ]
    )

    # Create augmented images
    augmented_images = [augmentations(image) for _ in range(n_augmentations)]

    # Display original and augmented images
    fig, axes = plt.subplots(1, n_augmentations + 1, figsize=figsize)

    # Display original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Display augmented images
    for i, aug_img in enumerate(augmented_images):
        axes[i + 1].imshow(aug_img)
        axes[i + 1].set_title(f"Augmentation {i+1}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_dataset_statistics(
    X_train, y_train, X_valid, y_valid, X_test, y_test, class_names
):
    """
    Visualize dataset statistics

    Args:
        X_train, y_train, X_valid, y_valid, X_test, y_test: Data and labels
        class_names (dict): Dictionary mapping class indices to names
    """
    print("\nDrawing data visualization charts...")

    # 1. Display examples for each class
    visualize_class_examples(X_train, y_train, class_names, n_samples=5)

    # 2. Display class distribution
    n_classes = len(set(y_train))

    # Calculate samples per class
    class_counts = {"class": [], "name": [], "train": [], "valid": [], "test": []}

    for i in range(n_classes):
        train_count = np.sum(y_train == i)
        valid_count = np.sum(y_valid == i) if y_valid is not None else 0
        test_count = np.sum(y_test == i) if y_test is not None else 0

        class_counts["class"].append(i)
        class_counts["name"].append(class_names[i])
        class_counts["train"].append(train_count)
        class_counts["valid"].append(valid_count)
        class_counts["test"].append(test_count)

    plt.figure(figsize=(12, 5))
    df = pd.DataFrame(class_counts)

    # Training set distribution
    plt.subplot(1, 3, 1)
    sns.barplot(x="class", y="train", data=df)
    plt.title("Training Set Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.xticks(
        df["class"],
        [f"{i}\n{name}" for i, name in zip(df["class"], df["name"])],
        rotation=45,
    )

    # Validation set distribution
    plt.subplot(1, 3, 2)
    sns.barplot(x="class", y="valid", data=df)
    plt.title("Validation Set Distribution")
    plt.xlabel("Class")
    plt.xticks(
        df["class"],
        [f"{i}\n{name}" for i, name in zip(df["class"], df["name"])],
        rotation=45,
    )

    # Test set distribution
    plt.subplot(1, 3, 3)
    sns.barplot(x="class", y="test", data=df)
    plt.title("Test Set Distribution")
    plt.xlabel("Class")
    plt.xticks(
        df["class"],
        [f"{i}\n{name}" for i, name in zip(df["class"], df["name"])],
        rotation=45,
    )

    plt.tight_layout()
    plt.show()

    # 3. Display image intensity distribution
    visualize_image_intensity(X_train)

    # 4. Display image size distribution
    if len(X_train) > 0:
        heights = [img.shape[0] for img in X_train]
        widths = [img.shape[1] for img in X_train]

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(heights, bins=20)
        plt.title("Image Height Distribution")
        plt.xlabel("Height (pixels)")
        plt.ylabel("Number of Images")

        plt.subplot(1, 2, 2)
        plt.hist(widths, bins=20)
        plt.title("Image Width Distribution")
        plt.xlabel("Width (pixels)")

        plt.tight_layout()
        plt.show()

    print("Dataset visualization complete.")


def visualize_training_results(
    train_loss_list,
    train_acc_list,
    val_loss_list,
    val_acc_list,
    best_acc,
    optimizer,
    learning_rate,
    batch_size,
    scheduler_type,
):
    """
    Visualize training results

    Args:
        train_loss_list (list): Training loss history
        train_acc_list (list): Training accuracy history
        val_loss_list (list): Validation loss history
        val_acc_list (list): Validation accuracy history
        best_acc (float): Best validation accuracy
        optimizer (str): Optimizer name
        learning_rate (float): Learning rate
        batch_size (int): Batch size
        scheduler_type (str): Scheduler type
    """
    plt.figure(figsize=(12, 10))

    # Training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(train_loss_list, label="Train")
    plt.plot(val_loss_list, label="Validation")
    plt.title("Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Training and validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(train_acc_list, label="Train")
    plt.plot(val_acc_list, label="Validation")
    plt.title("Accuracy Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    # Add best accuracy and other information
    plt.subplot(2, 2, 3)
    plt.axis("off")
    info_text = (
        f"Training Information:\n"
        f"Best Validation Accuracy: {best_acc:.2f}%\n"
        f"Optimizer: {optimizer}\n"
        f"Learning Rate: {learning_rate}\n"
        f"Batch Size: {batch_size}\n"
        f"Learning Rate Scheduler: {scheduler_type}\n"
    )
    plt.text(0.1, 0.5, info_text, fontsize=10)

    plt.tight_layout()

    # Save the chart
    os.makedirs("results", exist_ok=True)
    plt.savefig(os.path.join("results", "training_results.png"))
    plt.show()


def visualize_predictions(model, test_loader, class_names, device, num_samples=16):
    """
    Visualize predictions

    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        class_names (dict): Dictionary mapping class indices to names
        device (torch.device): Device to run the model on
        num_samples (int): Number of samples to visualize
    """
    model.eval()

    # Get a batch of data
    try:
        data_iter = iter(test_loader)
        batch = next(data_iter)
    except StopIteration:
        print("Test data is empty")
        return

    # Get images and labels 
    if len(batch) > 2:  # Dataset contains original images
        _, orig_images, targets = batch
        inputs = orig_images  # Use original images for visualization
    else:
        inputs, targets = batch

    # Limit the number of samples
    num_samples = min(num_samples, inputs.size(0))

    # Create grid layout
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # Get predictions
    # TODO get predcitions 
    outputs = model(inputs.to(device))
    _, preds = torch.max(outputs, 1)

    # Display images and predictions
    for i in range(num_samples):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i]

        # Get image
        img = inputs[i].permute(1, 2, 0).cpu().numpy()

        # Display image
        ax.imshow(img)

        # Set title
        pred_class = class_names[preds[i].item()]
        true_class = class_names[targets[i].item()]

        if preds[i] == targets[i]:
            color = "green"
        else:
            color = "red"

        ax.set_title(f"Pred: {pred_class}\nTrue: {true_class}", color=color)
        ax.axis("off")

    # Remove empty subplots
    for i in range(num_samples, rows * cols):
        if rows > 1:
            fig.delaxes(axes[i // cols, i % cols])
        else:
            try:
                fig.delaxes(axes[i])
            except:
                pass

    plt.tight_layout()

    # Save the image
    os.makedirs("results", exist_ok=True)
    plt.savefig(os.path.join("results", "predictions.png"))
    plt.show()


def plot_confusion_matrix(model, test_loader, class_names, device, criterion):
    """
    Plot confusion matrix

    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        class_names (dict): Dictionary mapping class indices to names
        device (torch.device): Device to run the model on
        criterion: Loss criterion

    Returns:
        dict: Dictionary containing test results
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    # Store predictions and true labels
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # If the dataset contains original images, ignore them
            if isinstance(inputs, list) and len(inputs) > 1:
                inputs = inputs[0]

            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Statistics
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Store predictions and targets
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate average loss and accuracy
    avg_loss = test_loss / len(test_loader)
    acc = 100.0 * correct / total

    print(f"Test Results: Loss: {avg_loss:.4f} | Acc: {acc:.2f}% ({correct}/{total})")

    # Calculate confusion matrix
    # TODO create confusion matrix
    cm = confusion_matrix(all_targets, all_preds) 

    # Print classification report
    class_names_list = [class_names[i] for i in range(len(class_names))]
    print("\nClassification Report:")
    # TODO Implement classification report 
    print(classification_report(all_targets, all_preds, target_names=class_names_list))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names_list,
        yticklabels=class_names_list,
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix")

    # Save confusion matrix
    os.makedirs("results", exist_ok=True)
    plt.savefig(os.path.join("results", "confusion_matrix.png"))
    plt.show()

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "targets": all_targets,
    }


def analyze_dataset(
    X_train, y_train, X_valid, y_valid, X_test, y_test, class_names, visualize=True
):
    """
    Analyze dataset and print statistics

    Args:
        X_train, y_train, X_valid, y_valid, X_test, y_test: Data and labels
        class_names (dict): Dictionary mapping class indices to names
        visualize (bool): Whether to create visualization charts
    """
    # Check if data is loaded
    if X_train is None or y_train is None:
        print("Error: No data provided for analysis.")
        return

    print("\n======== Dataset Statistics ========")
    n_train = len(X_train)
    n_validation = len(X_valid) if X_valid is not None else 0
    n_test = len(X_test) if X_test is not None else 0
    image_shape = X_train[0].shape if n_train > 0 else "N/A"
    n_classes = len(set(y_train))

    # Print basic information
    print(f"Number of training samples: {n_train}")
    print(f"Number of validation samples: {n_validation}")
    print(f"Number of test samples: {n_test}")
    print(f"Image shape: {image_shape} (height x width x channels)")
    print(f"Number of classes: {n_classes}")
    print("-" * 40)

    # Calculate samples per class
    class_counts = {"class": [], "name": [], "train": [], "valid": [], "test": []}

    for i in range(n_classes):
        train_count = np.sum(y_train == i)
        valid_count = np.sum(y_valid == i) if y_valid is not None else 0
        test_count = np.sum(y_test == i) if y_test is not None else 0

        class_counts["class"].append(i)
        class_counts["name"].append(class_names[i])
        class_counts["train"].append(train_count)
        class_counts["valid"].append(valid_count)
        class_counts["test"].append(test_count)

        print(
            f"Class {i} ({class_names[i]}): Training={train_count}, Validation={valid_count}, Test={test_count}"
        )

    print("-" * 40)
    print(f"Total: Training={n_train}, Validation={n_validation}, Test={n_test}")

    # Visualization
    if visualize:
        visualize_dataset_statistics(
            X_train, y_train, X_valid, y_valid, X_test, y_test, class_names
        )
