import rosbag
import rospy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# Path to your Rosbag file
bag_file = '/home/mathis_ros/Rosbags/Rosbags_sim/2024-05-13-11-41-54.bag'

# Topic name where the Vector3Stamped messages are logged
topic_name = '/err_from_img'

# Lists to store x, y, z coordinates, and timestamps
x_coords = []
y_coords = []
z_coords = []
t_stamps = []

# Read the Rosbag file
bag = rosbag.Bag(bag_file)

# Iterate over the messages in the specified topic
for topic, msg, t in bag.read_messages(topics=[topic_name]):
    x = msg.vector.x
    y = msg.vector.y
    z = msg.vector.z
    t = msg.header.stamp
    
    x_coords.append(x)
    y_coords.append(y)
    z_coords.append(z)
    t_stamps.append(t)

# Close the Rosbag file
bag.close()

# Specifying the starting point for plotting based on time percentage
start_percent = 70  # Start plotting from 70% of the total time span
start_time = t_stamps[0] + (t_stamps[-1] - t_stamps[0]) * start_percent / 100

# Filter data to start from the specified start time
filtered_indices = [i for i, t in enumerate(t_stamps) if t > start_time]
x_coords_cut = [x_coords[i] for i in filtered_indices]
y_coords_cut = [y_coords[i] for i in filtered_indices]
z_coords_cut = [z_coords[i] for i in filtered_indices]

# Create a 3D plot of the x, y, and z coordinates
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_coords_cut, y_coords_cut, z_coords_cut, linewidth=2.5)
ax.set_xlabel('X Coordinate', fontsize=26)
ax.set_ylabel('Y Coordinate', fontsize=26)
ax.set_zlabel('Z Coordinate', fontsize=26)
#ax.set_title('3D Path Taken', fontsize=22)
ax.tick_params(axis='both', which='major', labelsize=24)
plt.grid(True, linestyle='-', linewidth=1.5)  # Adding the grid with specific linestyle and linewidth
plt.savefig('Documentation/Images/finished/3d_path_plot_sim.png', bbox_inches='tight')

# Enable LaTeX rendering
plt.rc('text', usetex=True)
# Create a plot of the distance in x and y to the target
fig_2d = plt.figure(figsize=(10, 8))
plt.plot(x_coords_cut, y_coords_cut, 'k', linewidth=2.5, label='Trajectory')
plt.scatter([x_coords_cut[0]], [y_coords_cut[0]], c='k', marker='x', s=100, label='Start')
plt.scatter([x_coords_cut[-1]], [y_coords_cut[-1]], c='k', marker='o', s=100, label='End')
plt.xlabel(r'$x$ in m', fontsize=26)
plt.ylabel(r'$y$ in m', fontsize=26)
#plt.title('Distance to Target', fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.grid(True, linestyle='-', linewidth=1.5)  # Adding the grid with specific linestyle and linewidth
plt.legend(fontsize=22)
plt.savefig('Documentation/Images/finished/2d_plot_sim.png', bbox_inches='tight')


# Compute the distance traveled in the x-y plane as the cumulative sum of distances between consecutive points
distances = []
for i in range(0, len(x_coords_cut)):
    distance_temp = np.sqrt(x_coords_cut[i]**2 + y_coords_cut[i]**2)
    distances.append(distance_temp)

# Create the plot
fig, ax = plt.figure(figsize=(4, 12)), plt.gca()
ax.plot(distances, z_coords_cut, 'k-', linewidth=2.5, label='Trajectory')
ax.scatter([distances[0]], [z_coords_cut[0]], c='k', marker='x', s=100, label='Start')
ax.scatter([distances[-1]], [z_coords_cut[-1]], c='k', marker='o', s=100, label='End')

# Labeling the plot
ax.set_xlabel('Distance on XY Plane in m', fontsize=26)
ax.set_ylabel('Altitude in m', fontsize=26)
#ax.set_title('Altitude vs. Distance Traveled', fontsize=22)

# Set background and text colors for visibility
# ax.set_facecolor('k')
# fig.set_facecolor('k')
ax.tick_params(axis='both', which='major', labelsize=24)
plt.grid(True, linestyle='-', linewidth=1.5)  # Adding the grid with specific linestyle and linewidth
legend = ax.legend(fontsize=22)
# plt.setp(legend.get_texts(), color='w')
# legend.get_frame().set_facecolor('k')

# Save and show the plot
plt.savefig('Documentation/Images/finished/altitude_vs_distance_sim.png', bbox_inches='tight')
plt.show()


