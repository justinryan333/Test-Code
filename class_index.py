import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import matplotlib.pyplot as plt
import numpy as np

# Define a transformation to normalize the images
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a tensor
])

# Load the CIFAR-100 dataset
cifar100 = CIFAR100(root='./data', train=True, download=True, transform=transform)

# Get the list of class names from the dataset
class_names = cifar100.classes

# Specify the desired classes and the number of images per class
# Format: {class_index: num_images}
class_requests = {
    21: 1,  # 1 image from class 1
    22: 1,  # 1 image from class 13
    23: 1,  # 1 image from class 14
    24: 1,  # 1 image from class 15
    25: 1,  # 1 image from class 16
    26: 1,  # 1 image from class 17
    27: 1,  # 1 image from class 18
    28: 1,  # 1 image from class 19
    29: 1,  # 1 image from class 20
    30: 1  # 1 image from class 21
}

random_selection = True  # Change to False for non-random selection

# Prepare to collect selected indices and images
selected_indices = []
num_images_to_retrieve = []

# Loop through the specified classes and their requested image counts
for class_index, num_images in class_requests.items():
    # Get indices of images that belong to the specified class
    class_indices = [i for i, (_, label) in enumerate(cifar100) if label == class_index]

    if len(class_indices) < num_images:
        print(
            f"Not enough images available for class {class_index}. Available: {len(class_indices)}, Requested: {num_images}")
        num_images = len(class_indices)  # Adjust to the maximum available if not enough

    # Select images based on the random_selection variable
    if random_selection:
        selected = np.random.choice(class_indices, num_images, replace=False)
    else:
        selected = class_indices[:num_images]

    selected_indices.extend(selected)  # Add selected indices to the list
    num_images_to_retrieve.append(num_images)  # Store the requested number of images

# Create a figure for displaying the images
num_selected_images = len(selected_indices)
fig, axes = plt.subplots(1, num_selected_images, figsize=(15, 5))

# Loop through the selected indices and display the images
for ax, index in zip(axes, selected_indices):
    image, label = cifar100[index]

    # Convert the image from a tensor to a numpy array
    image = image.numpy().transpose(1, 2, 0)  # Change the order to (H, W, C)

    # Unnormalize the image (if you had normalized it)
    image = (image * 255).astype(np.uint8)

    # Display the image
    ax.imshow(image)

    # Set the title as the class name
    class_name = class_names[label]
    ax.set_title(f'{class_name}', fontsize=10)
    ax.axis('off')  # Hide axis

# Show the images
plt.tight_layout()
plt.show()
