import matplotlib.pyplot as plt

y = [3067, 2032, 1799, 1662, 1525, 1378]
x = [1, 0.9, 0.8, 0.7, 0.6, 0.5]
plt.xlim([1, 0.5])
plt.title('#proteins-sequence identity')
plt.ylabel('number clusters')
plt.xlabel('sequence identity')
plt.plot(x, y)
plt.savefig("proteins-sequence_id.pdf")
