import rosbag
import rospy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Path to your Rosbag folder
path_to_folder = '/home/mathis_ros/Rosbags/Rosbags_sim/'
#file_names = ['circle1.bag', 'circle2.bag', 'circle3.bag', 'circle5.bag']
file_names = ['tins.bag']

# Enable LaTeX rendering
plt.rc('text', usetex=True)

# Set common style properties
plt.rcParams.update({
    'font.size': 26,
    'axes.labelsize': 26,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 22,
    'lines.linewidth': 2.5,
    'grid.linewidth': 1.5
})

for file_name in file_names:
    bag_file = path_to_folder + file_name
    base_file_name = file_name.split('.')[0]

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

    # Specifying the starting point for plotting based on z_coord not being zero
    start_index = next((i for i, z in enumerate(z_coords) if z >= 11), None)
    if start_index is None:
        raise ValueError("All z_coords are zero, cannot start plot.")

    # Filter data to start from the specified start time
    x_coords_cut = x_coords[start_index:]
    y_coords_cut = y_coords[start_index:]
    z_coords_cut = z_coords[start_index:]
    t_stamps_cut = t_stamps[start_index:]

    # Compute the distance traveled in the x-y plane as the cumulative sum of distances between consecutive points
    distances = []
    for i in range(0, len(x_coords_cut)):
        distance_temp = np.sqrt(x_coords_cut[i] ** 2 + y_coords_cut[i] ** 2)
        distances.append(distance_temp)

    # Create the plot
    #5.5 or 9.25
    fig, ax = plt.subplots(figsize=(7.33, 10))
    ax.plot(distances, z_coords_cut, 'k-', label='Trajectory')
    ax.scatter([distances[0]], [z_coords_cut[0]], c='k', marker='x', s=100, label='Start')
    ax.scatter([distances[-1]], [z_coords_cut[-1]], c='k', marker='o', s=100, label='End')

    # Labeling the plot
    ax.set_xlabel('Distance on XY Plane in m')
    ax.set_ylabel('Altitude in m')
    # ax.set_title('Altitude vs. Distance Traveled')

    # Set x-axis limits
    ax.set_xlim([0, 2])

    # Add vertical lines
    ax.axvline(x=1, color='k', linestyle='--', label='Edge of platform (1m)')
    ax.axvline(x=0.36, color='k', linestyle=':', label='Outer circle (0.36m)')

    ax.tick_params(axis='both', which='major')
    ax.grid(True, linestyle='-')
    ax.legend()

    # Save and show the plot
    plt.savefig(f'Documentation/Images/finished/altitude_vs_distance_sim_{base_file_name}.png', bbox_inches='tight')
    plt.close(fig)

    # Convert timestamps to seconds for easier plotting
    t_seconds = [(t.to_sec() - t_stamps_cut[0].to_sec()) for t in t_stamps_cut]

    # Create altitude vs. time plot
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot(t_seconds, z_coords_cut, 'k-', label='Altitude')
    ax1.set_xlabel(r'$t$ in s')
    ax1.set_ylabel(r'Altitude in m')
    # Add vertical lines
    ax1.axvline(x=15.2, color='k', linestyle='--', label='')
    ax1.axvline(x=21.3, color='k', linestyle='--', label='')

    #Place text
    ax1.text(5, 11, 'Detecting square', fontsize=26, color='black')
    #Rotate text
    ax1.text(17.5, 5.7, 'Detecting circle', fontsize=26, color='black', rotation=-90)
    ax1.text(25.2, 4, 'Hovering and aligning', fontsize=26, color='black', rotation=-90)

    ax1.tick_params(axis='both', which='major')
    ax1.grid(True, linestyle='-')
    ax1.set_xlim([0, max(t_seconds)])
    #ax.axvline(x=21.5, color='k', linestyle=':', label='')
    #ax1.legend()

    plt.savefig(f'Documentation/Images/finished/altitude_vs_time_plot_sim_{base_file_name}.png', bbox_inches='tight')
    plt.close(fig)

    # Calculate error as distance from origin (0, 0, 0)
    errors_cut = [np.sqrt(x ** 2 + y ** 2) for x, y in zip(x_coords_cut, y_coords_cut)]

    # Create error vs. time plot
    fig, ax2 = plt.subplots(figsize=(10, 8))
    ax2.plot(t_seconds, errors_cut, 'k-', label='Error')
    ax2.set_xlabel(r'$t$ in s')
    ax2.set_ylabel(r'Horizontal distance in m')
    ax2.tick_params(axis='both', which='major')
    ax2.grid(True, linestyle='-')
    ax2.set_xlim([0, max(t_seconds)])
    ax2.set_ylim([0, 0.6])
    #ax2.legend()

    # Add vertical lines
    ax2.axvline(x=25, color='k', linestyle='--', label='')
    ax2.axvline(x=39.5, color='k', linestyle='--', label='')

    #Place text
    ax2.text(9, 0.51, 'Descending', fontsize=26, color='black')
    #Rotate text
    ax2.text(31, 0.21, 'Aligning over center circle', fontsize=26, color='black', rotation=-90)
    ax2.text(45, 0.335, 'Aligning over tin', fontsize=26, color='black', rotation=-90)

    plt.savefig(f'Documentation/Images/finished/error_vs_time_plot_sim_{base_file_name}.png', bbox_inches='tight')
    plt.close(fig)
