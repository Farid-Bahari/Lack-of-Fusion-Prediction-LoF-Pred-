import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit page configuration
st.set_page_config(page_title="Melt Pool Cross-Section Map", layout="centered")
st.title("Lack-of-Fusion Prediction in LPBF")

# Sidebar for user input
st.sidebar.header("Simulation Parameters")
width = st.sidebar.number_input("Melt Pool Width (µm)", value=138, min_value=10, max_value=1000, step=5)
depth = st.sidebar.number_input("Melt Pool Depth (µm)", value=69, min_value=5, max_value=1000, step=5)
layer_thickness = st.sidebar.number_input("Layer Thickness (µm)", value=25, min_value=1, max_value=500, step=1)
hatch_distance = st.sidebar.number_input("Hatch Distance (µm)", value=130, min_value=10, max_value=1000, step=5)
rotation_angle_deg = st.sidebar.number_input("Rotation Angle (degrees)", value=67, min_value=0, max_value=180, step=1)
cut_plane_depth = st.sidebar.number_input("Cut Plane Depth (µm)", value=350, min_value=0, max_value=5000, step=10)


# Fixed parameters
theta = np.linspace(0, np.pi, 100)
extrusion_depth = 1300
num_paths = 10
num_layers = 30
rotation_center = np.array([350, 650])

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

for j in range(num_layers):
    angle_rad = np.radians(j * rotation_angle_deg)
    Rz = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    extrude_dir = Rz @ np.array([0, 1])

    for i in range(num_paths):
        center_x = width + i * hatch_distance - 300
        center_y = 0
        center_z = j * layer_thickness

        # Half-ellipse + flat top
        x_arc = width/2 * np.cos(theta)
        z_arc = -depth * np.sin(theta)
        x = np.concatenate([x_arc, x_arc[::-1]])
        z = np.concatenate([z_arc, np.full_like(z_arc, z_arc[-1])])

        # Rotate in XY plane
        ellipse_xy = np.vstack([x + center_x, np.zeros_like(x) + center_y])
        ellipse_xy_centered = ellipse_xy - rotation_center[:, np.newaxis]
        ellipse_xy_rot = Rz @ ellipse_xy_centered + rotation_center[:, np.newaxis]

        x_rot = ellipse_xy_rot[0]
        y_rot = ellipse_xy_rot[1]
        z_rot = z + center_z

        # Extrude shape
        X0, Y0, Z0 = x_rot, y_rot, z_rot
        X1 = x_rot + extrusion_depth * extrude_dir[0]
        Y1 = y_rot + extrusion_depth * extrude_dir[1]
        Z1 = z_rot

        x_cross, z_cross = [], []

        # Front face intersection
        for k in range(len(X0) - 1):
            if (Y0[k] - cut_plane_depth) * (Y0[k+1] - cut_plane_depth) <= 0:
                if Y0[k] != Y0[k+1]:
                    t = (cut_plane_depth - Y0[k]) / (Y0[k+1] - Y0[k])
                    x_cross.append(X0[k] + t * (X0[k+1] - X0[k]))
                    z_cross.append(Z0[k] + t * (Z0[k+1] - Z0[k]))

        # Back face intersection
        for k in range(len(X1) - 1):
            if (Y1[k] - cut_plane_depth) * (Y1[k+1] - cut_plane_depth) <= 0:
                if Y1[k] != Y1[k+1]:
                    t = (cut_plane_depth - Y1[k]) / (Y1[k+1] - Y1[k])
                    x_cross.append(X1[k] + t * (X1[k+1] - X1[k]))
                    z_cross.append(Z1[k] + t * (Z1[k+1] - Z1[k]))

        # Side walls intersection
        for k in range(len(X0)):
            if (Y0[k] - cut_plane_depth) * (Y1[k] - cut_plane_depth) <= 0:
                if Y0[k] != Y1[k]:
                    t = (cut_plane_depth - Y0[k]) / (Y1[k] - Y0[k])
                    x_cross.append(X0[k] + t * (X1[k] - X0[k]))
                    z_cross.append(Z0[k] + t * (Z1[k] - Z0[k]))

        if x_cross:
            x_cross = np.array(x_cross)
            z_cross = np.array(z_cross)
            center_x_cross = np.mean(x_cross)
            center_z_cross = np.mean(z_cross)
            angles = np.arctan2(z_cross - center_z_cross, x_cross - center_x_cross)
            sorted_indices = np.argsort(angles)
            ax.fill(x_cross[sorted_indices], z_cross[sorted_indices], color='gray', edgecolor='black', linewidth=0.5)

# Axis limits
ax.set_xlim(0, 500)
ax.set_ylim(0, 500)

# Labels
ax.set_xticks([])  # remove x-axis ticks
ax.set_yticks([])  # remove z-axis ticks
ax.set_xlabel('')  # remove x-axis label
ax.set_ylabel('')

# Title with all parameters
ax.set_title(f"Cross-Section at y = {cut_plane_depth} µm | MPW = {width} µm, MPD = {depth} µm, "
             f"LT = {layer_thickness} µm, HD = {hatch_distance} µm, RA = {rotation_angle_deg}°")

# Aspect ratio
ax.set_aspect('equal', adjustable='box')

# Add scale bar (100 µm) at lower-right
scalebar_length = 100
x_start = 350
z_start = 20
ax.plot([x_start, x_start + scalebar_length], [z_start, z_start], color='white', linewidth=4)
ax.text(x_start + scalebar_length/2, z_start + 10, '100 µm', color='white', ha='center', va='bottom', fontsize=20)

plt.tight_layout()

# Display in Streamlit
st.pyplot(fig)
