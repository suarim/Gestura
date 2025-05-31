import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

# Read the Parquet file
df = pd.read_parquet('./static/1014384002.parquet')

# Filter out rows with NaN values in the x, y columns
df = df.dropna(subset=['x', 'y'])

# Create a directory to store frames
os.makedirs('frames', exist_ok=True)

# Define colors for different types of landmarks
colors = {
    'face': 'blue',
    'left_hand': 'green',
    'right_hand': 'red',
    'pose': 'black'
}

# Define connections for face and hands (adjust based on actual landmark indices)
connections = {
    'face': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
             (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17)],  # Example connections for face
    'left_hand': [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                  (10, 11), (11, 12), (5, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)],  # Example connections for left hand
    'right_hand': [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                   (10, 11), (11, 12), (5, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]  # Example connections for right hand
 ,'pose': [(11, 12), (11, 13), (12, 14),(14, 16), (16, 18), (13, 15),(11, 23), (12, 24), (23, 24), (15, 17)]  # Example connections for pose
}

# Generate and save each frame as an image
for frame_number in df['frame'].unique():
    frame_data = df[df['frame'] == frame_number]
    plt.figure(figsize=(6, 6))

    for landmark_type in frame_data['type'].unique():
        type_data = frame_data[frame_data['type'] == landmark_type]
        plt.scatter(type_data['x'], -type_data['y'], color=colors.get(landmark_type, 'black'), label=landmark_type)
        
        # Draw lines for connections
        for connection in connections.get(landmark_type, []):
            point1 = type_data[type_data['landmark_index'] == connection[0]]
            point2 = type_data[type_data['landmark_index'] == connection[1]]
            if not point1.empty and not point2.empty:
                plt.plot([point1['x'].values[0], point2['x'].values[0]], 
                         [-point1['y'].values[0], -point2['y'].values[0]], 
                         color=colors.get(landmark_type, 'black'))

    plt.xlim(0, 1)  # Adjust based on your data range
    plt.ylim(-1, 0)  # Inverted y-axis
    plt.title(f'Frame {frame_number}')
    plt.legend()
    plt.savefig(f'frames/frame_{frame_number:04d}.png')
    plt.close()

# Create a video from the saved frames
frame_folder = 'frames'
video_name = 'landmarks_animation.mp4'

images = [img for img in sorted(os.listdir(frame_folder)) if img.endswith(".png")]
frame = cv2.imread(os.path.join(frame_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(frame_folder, image)))

cv2.destroyAllWindows()
video.release()

# Clean up by removing the frames directory
import shutil
try:
    shutil.rmtree('frames')
    print("Successfully removed temporary frames directory.")
except PermissionError:
    print("\nWarning: Could not automatically remove the 'frames' directory due to a permission error.")
    print("It might still be in use. Please delete it manually after the script finishes.")
except Exception as e:
    print(f"\nAn error occurred during cleanup: {e}")

print("\nVideo creation process finished.")

