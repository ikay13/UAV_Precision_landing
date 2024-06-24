import matplotlib.pyplot as plt

# Set common style properties without LaTeX
plt.rcParams.update({
    'font.size': 26,
    'axes.labelsize': 26,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 22,
    'lines.linewidth': 2.5,
    'grid.linewidth': 1.5,
    'text.usetex': False  # Ensure LaTeX is not used
})

# Data points
x_coords_all = [11, 22, 22, -4, 8, 18, 5, 40, 3, 24]
y_coords_all = [26, -22, -12, -5.5, -8, -5, 1, 20, 12, 2]

x_avg = sum(x_coords_all) / len(x_coords_all)
y_avg = sum(y_coords_all) / len(y_coords_all)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 10))

# Draw the zero lines first
ax.axhline(0, color='black', linewidth=2)
ax.axvline(0, color='black', linewidth=2)

# Draw grid lines
ax.grid(True, linestyle='-')

# Plot data points on top of grid lines
ax.scatter(x_coords_all, y_coords_all, c='k', marker='o', zorder=5, label='Landing points')

# Annotating the points
for i, (x, y) in enumerate(zip(x_coords_all, y_coords_all), start=1):
    ax.annotate(f'{i}', (x, y), textcoords="offset points", xytext=(18, -5), ha='center', zorder=10)

#Add avg center
ax.scatter(x_avg, y_avg, c='k', marker='x', s=200 ,zorder=5, label='Average landing point')

# Setting labels and title
ax.set_xlabel('Distance eastward in cm')
ax.set_ylabel('Distance northward in cm')

# Set x and y axis limits to match original data range
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)

#Direction
dx = -20
dy = -4.14*2
startpoint = [-21, 21]
endpoint = [startpoint[0] + dx, startpoint[1] + dy]
ax.arrow(startpoint[0], startpoint[1], dx, dy, head_width=2, head_length=3, fc='k', zorder = 15, width=0.5)
## Add text
ax.text(-42, 17, 'Wind', fontsize=26, color='black', rotation=22)

#legend
ax.legend()

# Show plot
plt.show()
