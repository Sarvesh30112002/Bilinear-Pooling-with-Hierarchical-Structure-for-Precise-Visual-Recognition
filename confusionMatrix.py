import numpy as np
import matplotlib.pyplot as plt
import itertools

# Define the confusion matrix
confusion_matrix = np.array(
    [
        [21, 1, 4, 0, 0, 0],
        [1, 22, 1, 0, 1, 0],
        [2, 1, 25, 0, 0, 0],
        [0, 0, 0, 19, 0, 0],
        [0, 0, 0, 0, 20, 0],
        [0, 0, 0, 0, 0, 16],
    ]
)

# Define the class names (optional)
class_names = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6"]

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

# Show class labels on x and y axes
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# Display values in each cell
thresh = confusion_matrix.max() / 2.0
for i, j in itertools.product(
    range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])
):
    plt.text(
        j,
        i,
        format(confusion_matrix[i, j], "d"),
        horizontalalignment="center",
        color="white" if confusion_matrix[i, j] > thresh else "black",
    )

plt.tight_layout()
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()
