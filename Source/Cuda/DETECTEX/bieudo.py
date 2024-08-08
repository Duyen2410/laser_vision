import matplotlib.pyplot as plt
import numpy as np

# Data before optimization
labels = ['Ransac', 'LLS', 'Hough Transform']
time_algorithm_before = [46.91, 0.001, 8.7415]
time_calibration_before = [54.91, 8.001, 16.7415]
r_square_before = [0.9081, 0.9111, 0.8927]

# Data after second optimization
time_algorithm_after = [0.1392, 0.001, 0.4411]
time_calibration_after = [5.7392, 5.601, 6.041]
r_square_after = [0.9111, 0.9111, 0.3219]

# Calculate improvement ratios
improvement_algorithm = [(before - after) / before if before != 0 else 0 for before, after in zip(time_algorithm_before, time_algorithm_after)]
improvement_calibration = [(before - after) / before if before != 0 else 0 for before, after in zip(time_calibration_before, time_calibration_after)]
r_square_change = [after - before for before, after in zip(r_square_before, r_square_after)]

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

x = np.arange(len(labels))
width = 0.35

# Algorithm time improvement
axs[0].bar(x, improvement_algorithm, width, label='Algorithm Time Improvement', color='blue', alpha=0.7)
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels)
axs[0].set_title('Algorithm Time Improvement Ratio')
axs[0].set_ylabel('Improvement Ratio')

# Calibration time improvement
axs[1].bar(x, improvement_calibration, width, label='Calibration Time Improvement', color='green', alpha=0.7)
axs[1].set_xticks(x)
axs[1].set_xticklabels(labels)
axs[1].set_title('Calibration Time Improvement Ratio')
axs[1].set_ylabel('Improvement Ratio')

# R-square change
axs[2].bar(x, r_square_change, width, label='R-square Change', color='red', alpha=0.7)
axs[2].set_xticks(x)
axs[2].set_xticklabels(labels)
axs[2].set_title('R-square Change')
axs[2].set_ylabel('Change in R-square')

plt.tight_layout()
plt.show()
