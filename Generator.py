import numpy as np
import matplotlib.pyplot as plt

print("🔃 Generating the dataset")
from numpy import random

# dataset_1 = random.normal(loc=10, scale=1, size=(150, 3))
# dataset_2 = random.normal(loc=20, scale=2, size=(150, 3))
# dataset_3 = random.normal(loc=30, scale=3, size=(150, 3))
# dataset_4 = random.normal(loc=40, scale=4, size=(150, 3))

dataset_1 = random.randint(0,256,(150,3))
dataset_1[:, 1] = 0
dataset_1[:, 2] = 0

dataset_2 = random.randint(0,256,(150,3))
dataset_2[:, 0] = 0
dataset_2[:, 2] = 0

dataset_3 = random.randint(0,256,(150,3))
dataset_3[:, 0] = 0
dataset_3[:, 1] = 0

dataset_4 = random.randint(0,256,(150,3))
dataset_4[:, 0] = 0




np.savetxt('generated/dataset_1.csv', dataset_1, delimiter= ',')
np.savetxt('generated/dataset_3.csv', dataset_3, delimiter= ',')
np.savetxt('generated/dataset_2.csv', dataset_2, delimiter= ',')
np.savetxt('generated/dataset_4.csv', dataset_4, delimiter= ',')

dataset_all = np.vstack((dataset_1, dataset_2, dataset_3, dataset_4))
np.savetxt('generated/dataset_all.csv', dataset_all, delimiter= ',')

fig = plt.figure(figsize=(10,10))
tuple(dataset_all)
plt.imshow(dataset_all)
plt.title("Plot datasets")
plt.show()


print("✅ Dataset generation complete !")


