from sklearn.metrics import f1_score, confusion_matrix
from data import MyDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import hbp_model

# Define transformation for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Load the fine-tuned model
model = hbp_model.Net()
model.load_state_dict(torch.load("firststep.pth"))
model.eval()


# Load the test dataset
test_dataset = MyDataset("test_images_shuffle.txt", transform)

# Create a DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Make predictions on the test set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.numpy())
        predicted_labels.extend(predicted.numpy())

# Compute F1 score
f1 = f1_score(true_labels, predicted_labels, average='macro')

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Calculate error count
errors = sum(true != pred for true, pred in zip(true_labels, predicted_labels))

# Print results
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)
print("Total Errors:", errors)
