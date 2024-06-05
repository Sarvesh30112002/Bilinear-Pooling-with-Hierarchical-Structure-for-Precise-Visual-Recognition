import matplotlib.pyplot as plt

# Read loss values from the file
with open('loss.txt', 'r') as file:
    lines = file.readlines()

# Extract epoch numbers and loss values
epochs = []
losses = []
for line in lines:
    parts = line.strip().split(': ')
    if len(parts) != 2:
        print("Error parsing line:", line)
        continue
    epoch, loss = parts
    epochs.append(int(epoch.split()[1]))
    losses.append(float(loss))

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o', linestyle='-')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
