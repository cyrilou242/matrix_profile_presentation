import numpy as np
import matplotlib.pyplot as plt

# Generate time values
time = np.linspace(0, 10, 20)  # Adjust the time range and resolution as needed

# Generate two sinusoidal time series (you can adjust these equations)
series_A = np.sin(time)
# small distance
series_B = np.sin(time - 0.001) + 0.5  # Adjust the equation for Time Series B
# big distance - uncomment to generate
#series_B = np.cos(time - 0.001) + 0.5  # Adjust the equation for Time Series B

# Calculate Euclidean distances
euclidean_distances = np.sqrt((series_A - series_B) ** 2)

# Create the plot
plt.figure(figsize=(10, 6))
#plt.plot(time, series_A, label='Time Series A', color='blue')
#plt.plot(time, series_B, label='Time Series B', color='red')
plt.plot(time, series_A,  color='blue')
plt.plot(time, series_B,  color='red')
plt.xlabel('Time')
plt.ylabel('Amplitude')
#plt.title('Euclidean Distance Between Sinusoidal Time Series')
plt.legend()

# Add lines for Euclidean distances
for i in range(len(time)):
    plt.plot([time[i], time[i]], [series_A[i], series_B[i]], 'k--', lw=0.5)

# Show the Euclidean distance for a specific point (you can adjust the index)
#index_to_highlight = 10  # Adjust this index as needed
#plt.text(time[index_to_highlight], 0.8, f'Euclidean Distance: {euclidean_distances[index_to_highlight]:.2f}', fontsize=10)

# Display the plot
plt.grid(False)
plt.show()
