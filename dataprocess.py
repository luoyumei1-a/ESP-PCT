## dataprocess.py
import scipy.io as scio
import glob
import numpy as np
import os
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Global variables to save the bounding box of the previous frame
last_xmin = -0.3
last_xmax = 0.3
last_ymin = 1.1
last_ymax = 1.5
last_zmin = -0.2
last_zmax = 0.3

def is_hand_point(xyz):
    # Set the predefined range for the hand point cloud
    xmin = -0.3; xmax = 0.3
    ymin = 1.1; ymax = 1.5
    zmin = -0.2; zmax = 0.3
    # Find the hand point cloud
    hand_indices = (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) & \
                   (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax) & \
                   (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax)
    return hand_indices

def get_bounding_box(points):
    global last_xmin, last_xmax, last_ymin, last_ymax, last_zmin, last_zmax

    if points.size == 0:
        return last_xmin, last_xmax, last_ymin, last_ymax, last_zmin, last_zmax
    # Extract the x, y, and z coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # Compute the min and max for each dimension
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    zmin, zmax = np.min(z), np.max(z)
    last_xmin, last_xmax = xmin, xmax
    last_ymin, last_ymax = ymin, ymax
    last_zmin, last_zmax = zmin, zmax
    return xmin, xmax, ymin, ymax, zmin, zmax

list_all_ti = []  # ti数据
list_all_label = []  # body label
list_all_key2 = []  # joint point
rootdir = r".\matData"
Keyboard = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']
Repect = ['0','1','2','3','4','5','6']

for iKeyboard in range(0, len(Keyboard)):  # Iterate over each keyboard
    print(Keyboard[iKeyboard])
    for iRepect in range(0, len(Repect)):  # Iterate over each repetition
        subDir = os.path.join(rootdir, Repect[iRepect], Keyboard[iKeyboard])
        print(subDir)
        subDirlist = os.listdir(subDir)
        subDirlist.sort()
        list_person = []
        hand_points_history = []
        for j in range(0, len(subDirlist)):  # Iterate over each session's data
            path = os.path.join(subDir, subDirlist[j])
            if not os.path.isdir(path):
                continue
            matFilelist = glob.glob(os.path.join(path, "*.mat"))
            if len(matFilelist) < 25:
                continue
            
            matFilelist = sorted(matFilelist, key=lambda x: int(x.split('_')[-1].split('.')[0]))

            list_repeat_once_ti = []
            list_person_once_key2 = []
            
            for frame in range(0, len(matFilelist)):  # Iterate over each frame
                data = scio.loadmat(matFilelist[frame])
                pc_xyziv_ti = data['pc_xyziv_ti'][:, 0:5]  # Millimeter wave point cloud features dimension 0:5
                pc_xyz_key = data['pc_xyz_key'][:, 0:3]
                # pc_xyz_key = np.asarray(
                #     [pc_xyz_key0[0], pc_xyz_key0[1], pc_xyz_key0[2], pc_xyz_key0[3], pc_xyz_key0[5],
                #      pc_xyz_key0[6], pc_xyz_key0[7], pc_xyz_key0[12], pc_xyz_key0[13], pc_xyz_key0[14],
                #      pc_xyz_key0[18], pc_xyz_key0[19], pc_xyz_key0[20], pc_xyz_key0[22], pc_xyz_key0[23],
                #      pc_xyz_key0[24], pc_xyz_key0[26]])
                
                pc_xyz_key[:,2] = pc_xyz_key[:,2]-0.1
                pc_xyziv_ti[:,0] = pc_xyziv_ti[:,0]-0.2
                pc_xyziv_ti[:,1] = pc_xyziv_ti[:,1]+0.3
                pc_xyziv_ti[:,2] = pc_xyziv_ti[:,2]-0.9

                # Find hand point indices
                hand_indices = is_hand_point(pc_xyziv_ti)
                # Extract hand points using boolean indexing
                hand_points = pc_xyziv_ti[hand_indices, :]
                
                # If the number of hand point clouds is less than 6, and at least 2 frames have passed
                if hand_points.shape[0] < 6 and frame > 2 and len(hand_points_history) >= 2:
                    # Calculate the average position change of the hand point clouds over the past 10 frames
                    diffs = [np.diff(hand_points, axis=0) for hand_points in hand_points_history]
                    avg_change = np.mean([np.mean(diff, axis=0) for diff in diffs], axis=0)
                    # Use the average change to predict the current frame's hand point cloud position
                    hand_points = np.vstack((hand_points, hand_points_history[-1] + 3 * avg_change))
                    # Add the predicted hand point cloud to pc_xyziv_ti
                    pc_xyziv_ti = np.vstack((pc_xyziv_ti, hand_points))
                    
                # Re-identify points belonging to the hand, returning a logical array
                hand_indices = is_hand_point(pc_xyziv_ti)
                # Re-extract the hand point cloud using logical indexing
                hand_points = pc_xyziv_ti[hand_indices]
                # Calculate the cubic boundary of the hand point cloud
                xmin, xmax, ymin, ymax, zmin, zmax = get_bounding_box(hand_points)

                # Update the history of hand point clouds
                if len(hand_points_history) < 5:
                    hand_points_history.append(hand_points)
                else:
                    hand_points_history = hand_points_history[1:]
                    hand_points_history.append(hand_points)
                
                # Process bounding box into fixed number of point clouds ti96
                pc_frame_ti = np.zeros((96, 5), dtype=np.float32)  # Note the change here
                pc_num_ti = pc_xyziv_ti.shape[0]
                hand_point_num = hand_points.shape[0]
                        
                if hand_point_num > 8:              
                    if hand_point_num < 96:
                        fill_list = np.random.choice(96, size=hand_point_num, replace=False)
                        fill_set = set(fill_list)
                        pc_frame_ti[fill_list, :5] = hand_points  # Only fill the first four columns with hand_points
                        # pc_frame_ti[fill_list, 5:] = [xmin, xmax, ymin, ymax, zmin, zmax]  # Fill the rest with the bounding box info
                        dupl_list = [x for x in range(96) if x not in fill_set]
                        dupl_pc = np.random.choice(hand_point_num, size=len(dupl_list), replace=True)
                        pc_frame_ti[dupl_list, :5] = hand_points[dupl_pc]
                        # pc_frame_ti[dupl_list, 5:] = [xmin, xmax, ymin, ymax, zmin, zmax]  # Fill the rest with the bounding box info
                    else:
                        pc_list = np.random.choice(min(hand_point_num, 96), size=96, replace=False)
                        pc_frame_ti[pc_list, :5] = hand_points[pc_list]
                        # pc_frame_ti[pc_list, 5:] = [xmin, xmax, ymin, ymax, zmin, zmax]  # Fill the rest with the bounding box info
                elif pc_num_ti < 96:
                    fill_list = np.random.choice(96, size=pc_num_ti, replace=False)
                    fill_set = set(fill_list)
                    pc_frame_ti[fill_list, :5] = pc_xyziv_ti  # Only fill the first four columns with pc_xyziv_ti
                    # pc_frame_ti[fill_list, 5:] = [xmin, xmax, ymin, ymax, zmin, zmax]  # Fill the rest with the bounding box info
                    dupl_list = [x for x in range(96) if x not in fill_set]
                    dupl_pc = np.random.choice(pc_num_ti, size=len(dupl_list), replace=True)
                    pc_frame_ti[dupl_list, :5] = pc_xyziv_ti[dupl_pc]
                    # pc_frame_ti[dupl_list, 5:] = [xmin, xmax, ymin, ymax, zmin, zmax]  # Fill the rest with the bounding box info
                else:
                    pc_list = np.random.choice(min(pc_num_ti, 96), size=96, replace=False)
                    pc_frame_ti[pc_list, :5] = pc_xyziv_ti[pc_list]
                    # pc_frame_ti[pc_list, 5:] = [xmin, xmax, ymin, ymax, zmin, zmax]  # Fill the rest with the bounding box info

                # Identify rows containing NaN
                nan_rows = np.any(np.isnan(pc_frame_ti), axis=1)
                # Identify rows not containing NaN
                if hand_point_num > 0:
                    non_nan_rows = np.all(~np.isnan(hand_points), axis=1)
                else:
                    non_nan_rows = np.all(~np.isnan(pc_frame_ti), axis=1)

                # If there are rows containing NaN
                if np.any(nan_rows):
                    # Find all rows that do not contain NaN
                    if hand_point_num > 0:
                        valid_data = hand_points[non_nan_rows]
                    else:
                        valid_data = pc_frame_ti[non_nan_rows]
                    # Randomly select data from valid_data to replace rows with NaN
                    for i in range(pc_frame_ti.shape[0]):
                        if nan_rows[i]:
                            # Randomly choose an index
                            random_idx = np.random.choice(valid_data.shape[0])                            
                            # Replace NaN row with the chosen row
                            pc_frame_ti[i] = valid_data[random_idx]

                # Identify rows containing NaN
                nan_rows = np.any(np.isnan(pc_frame_ti), axis=1)
                # If there are rows containing NaN
                if np.any(nan_rows):
                    print('Exist NaN')

                list_repeat_once_ti.append(pc_frame_ti)
                list_person_once_key2.append(pc_xyz_key)

                # Process the number of frames to a consistent 25 frames
                list_repeat_once_ti_25 = list_repeat_once_ti
            list_repeat_once_key_25 = list_person_once_key2
            list_all_ti.append(list_repeat_once_ti_25)
            list_all_key2.append(list_repeat_once_key_25)
            list_all_label.append(Keyboard[iKeyboard])

print("data load end")
list_all_ti = np.asarray(list_all_ti)
nan_rows = np.any(np.isnan(list_all_ti), axis=1)
if np.any(nan_rows):
    print('Exist NaN')
list_all_label = np.asarray(list_all_label)
list_all_key2 = np.asarray(list_all_key2)

print(list_all_ti.shape)
print(list_all_label.shape)
print(list_all_key2.shape)

np.save(os.path.join(r".\npydata\train","list_all_ti"),list_all_ti)
np.save(os.path.join(r".\npydata\train","list_all_label"),list_all_label)
np.save(os.path.join(r".\npydata\train","list_all_key2"),list_all_key2)