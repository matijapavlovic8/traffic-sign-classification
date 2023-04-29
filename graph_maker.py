import os
import matplotlib.pyplot as plt

directory = './dataset/myData/myData'

subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

file_counts = []
for subdir in subdirs:
    subdir_path = os.path.join(directory, subdir)
    file_count = len([f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])
    file_counts.append(file_count)


labels = []
for i in range(1, 44):
    labels.append(i)

# Create the bar chart
plt.bar(labels, file_counts)

# Add a title and axis labels
plt.title('Dataset distribution')
plt.xlabel('Classes')
plt.ylabel('Number of images')

# Display the chart
plt.show()
