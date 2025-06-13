# -*- coding: utf-8 -*-
"""
Created on Fri May 16 15:08:10 2025

@author: Kari
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 15 01:32:31 2025

@author: marzu
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.filters import unsharp_mask
from skimage.feature import match_template, peak_local_max
import pandas as pd
import pims
import trackpy as tp
import imageio
import matplotlib as mpl
from skimage.color import rgba2rgb, rgb2gray  # For image color conversions
import cv2
from sklearn.neighbors import KDTree

# Define a function to do template matching and find peak matches
def MatchTemplate1(img, template1, thresh):
    # Compute a match score for how well the template matches each position in the image
    match1 = match_template(img, template1, pad_input=True)

    # Find peaks in the match score image that are local maxima and above a certain threshold
    peaks1 = peak_local_max(match1, min_distance=10, threshold_rel=thresh)

    return match1, peaks1  # Return both the match image and the list of peak positions

def MatchTemplate2(img, template2, thresh):
    # Compute a match score for how well the template matches each position in the image
    match2 = match_template(img, template2, pad_input=True)

    # Find peaks in the match score image that are local maxima and above a certain threshold
    peaks2 = peak_local_max(match2, min_distance=10, threshold_rel=thresh)

    return match2, peaks2  # Return both the match image and the list of peak positions


# Define an image preprocessing function using PIMS pipeline
@pims.pipeline
def preprocess_img(frame):
    
    frame = frame[0:1200, 0:1500]

    #the next few lines are to convert a RGBA image into a grayscale image
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = rgba2rgb(frame)
    if frame.ndim == 3 and frame.shape[2] == 3:
        frame = rgb2gray(frame)
    frame = (frame * 255).astype(np.uint8)

    # Sharpen the image using unsharp masking
    frame = unsharp_mask(frame, radius=2, amount=5)

    # Normalize the sharpened image back to 8-bit range
    frame *= 255.0 / frame.max()
    
    # Return the final preprocessed image
    return frame.astype(np.uint8)


# Function to plot trajectories up to a specific frame number `k`
def plotTraj(traj, k, directory, frames,COM):
    plt.figure(figsize=(12, 8), dpi=150)  # Create a figure
    plt.clf()  # Clear current figure (in case it’s being reused)

    # Filter the trajectory DataFrame to include only frames up to `k`
    subset = traj[traj.frame <= k]
    plt.xlim(700,1000)
    plt.ylim(500,900)
    # Plot the trajectories over the background image at frame `k`
    plt.scatter(COM_df.loc[COM_df['frame'] == k, 'x'].values[0], COM_df.loc[COM_df['frame'] == k, 'y'].values[0],
                color='red', marker='x', s=100, label='Center of Mass')
    plt.scatter(COM_df.loc[COM_df['frame'] == framei, 'x'].values[0], COM_df.loc[COM_df['frame'] == framei, 'y'].values[0], 
                color='dodgerblue', marker='x', s=75, label='Initial Center of Mass')

    tp.plot_traj(subset, colorby='particle', cmap=mpl.cm.winter,
                 superimpose=frames[k-framei], plot_style={'linewidth': 2})
    
    
#%%
def findNNdf(df, rad):
    '''
    This code uses a KD tree to quickly find how many points are within
    a given distance of each point's neighbor points. Should use rad>droplet radius
    to account for any errors in finding the position.
    The function returns the original DataFrame with added columns for nearest neighbor indices,
    number of nearest neighbors, and the fraction of nearest neighbors with a given number of neighbors.
    Parameters:
    df : pandas.DataFrame
        DataFrame containing the x and y coordinates along with any other columns
    rad : float
        The radius distance within which nearest neighbors are found
    Returns:
    df : pandas.DataFrame
        The original DataFrame with added columns for nearest neighbors information
    nn_df : pandas.DataFrame
        DataFrame containing the nearest neighbor indices, number of neighbors, and fraction of neighbors
    '''
    # Extract x and y coordinates from the DataFrame
    points = df[['x', 'y']].values
    # Build the KDTree with the given points
    tree = KDTree(points, leaf_size=2)
    # Find the nearest neighbors within the given radius
    nn_indices = tree.query_radius(points, r=rad)
    # Map the indices to particle numbers using the 'particle' column
    particle_numbers = df['particle'].values
    nn_particle_numbers = [particle_numbers[i] for i in nn_indices]
    # Calculate the number of nearest neighbors for each point (excluding itself)
    numNN = np.array([len(i) - 1 for i in nn_indices])  # Subtract 1 to exclude the point itself
    # Add the nearest neighbor data to the original DataFrame
    df['nearest_neighbors'] = nn_particle_numbers
    df.loc[:,'num_neighbors'] = numNN
    #return df, nn_fraction_df
    return df
def filter_neighbors_by_size(info,tolerance=1.1):
    '''
    This function updates the 'nearest_neighbors' column of the DataFrame by filtering the neighbors
    based on their distances and sizes. The distance between a particle and its neighbors is compared
    to the sum of their 'size' values. Only neighbors that are within this distance are retained.
    Parameters:
    df : pandas.DataFrame
        The DataFrame containing particle data, including 'x', 'y', 'size', and 'nearest_neighbors'
    x_col : str, optional
        The name of the column representing the x-coordinate (default is 'x')
    y_col : str, optional
        The name of the column representing the y-coordinate (default is 'y')
    size_col : str, optional
        The name of the column representing the particle's size (default is 'size')
    rad_col : str, optional
        The name of the column representing the nearest neighbors' particle numbers (default is 'nearest_neighbors')
    particle_col : str, optional
        The name of the column representing the particle identifier (default is 'particle')
    Returns:
    df : pandas.DataFrame
        The DataFrame with the updated 'nearest_neighbors' column, filtered by size-distance criteria
    '''
    df=info
    # Get the particle positions and sizes
    particles = df['particle'].values
    sizes = df['size'].values
    x_coords = df['x'].values
    y_coords = df['y'].values
    # Loop through each particle
    updated_neigh_list = []
    numneigh=[]

    
    for i, particle_id in enumerate(particles):
        # Get the list of neighboring particles' IDs
        df = df.reset_index(drop=True)
        neighbors = df[df['particle']==particle_id]['nearest_neighbors'].iloc[0]
        # print(neighbors)
        # Get the current particle's position and size
        x_i, y_i, size_i = x_coords[i], y_coords[i], sizes[i]
        # Initialize the list of valid neighbors for this particle
        valid_neighbors = []
        # Check each neighbor
        for neighbor_id in neighbors:
            #This will skip calling itself it's own neighbour
            if neighbor_id == particle_id:
                continue
            # Get the index of the neighbor
            neighbor_idx = df[df['particle'] == neighbor_id].index[0]
            # Get the neighbor's position and size
            x_j, y_j, size_j = x_coords[neighbor_idx], y_coords[neighbor_idx], sizes[neighbor_idx]
            # Calculate the distance between the particle and its neighbour
            distance = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
            # Check if the distance is less than or equal to the sum of their sizes
            if distance <= (size_i + size_j)*tolerance:
                #print('MAX DIS=' + str(distance))
                valid_neighbors.append(neighbor_id)
        # Append the valid neighbors' particle IDs to the updated list
        updated_neigh_list.append(valid_neighbors)

        numneigh.append(len(valid_neighbors))
    # Replace the 'nearest_neighbors' column with the filtered list of valid neighbors
    df['nearest_neighbors'] = updated_neigh_list
    df.loc[:,'num_neighbors'] = numneigh
    return df

def Centre_of_Mass(coords):
    '''Function to calculate the CofM position of a set of particles given list/array of positions'''
    totalx = np.mean(coords[:,0])
    totaly = np.mean(coords[:,1])
    return np.array([totalx,totaly])



#%% Section: Loading images and setup

# Set up the directory and file paths
directory = 'D:/Maya/'  # Folder where images are stored
run = '250612 Oval Droplet Shape 4/'  # Subfolder for this specific experiment
prefix = '*.tiff'  # File pattern to load TIFF images
#framei= 3524
#framef = 5194
#framei=1563
#framef=1650
# framef=1800
framei=1
framef=608

# Load a sequence of images and preprocess them
sequence = pims.ImageSequence(os.path.join(directory + run + prefix))
frames = preprocess_img(sequence[framei:framef])  # Preprocess the first 50 frames


#%% Section: Loop through all frames and detect peaks

# Initialize a list to store positions
poss = []
# Loop over each frame (except the last one)
for i in range(framei,framef-1):
    img=frames[i-framei]
    #blur = cv2.GaussianBlur(img, (5, 5), 0)
    blur=img
    # Invert if circles are dark on light background (optional, depending on contrast)
    inv = cv2.bitwise_not(blur)
    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,            # Inverse ratio of accumulator resolution to image resolution
        minDist=20,        # Minimum distance between circle centers
        param1=150,         # Higher threshold for Canny edge detector
        param2=35,         # Accumulator threshold for circle detection - one you want to change if circles are not being detected
        minRadius=10,       # Minimum circle radius
        maxRadius=60       # Maximum circle radius
    )
    # Draw circles
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        for x, y, r in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)      # Circle outline
            cv2.circle(output, (x, y), 2, (255, 255, 255), 3)      # Circle center
            poss.append([i, x,y,r])
    if (i%100==0):        
        print('Frame = ' + str(i))
    # Show result
        plt.imshow(output)
        plt.title("Hough Circle Detection")
        plt.xlim(500,1500)
        plt.ylim(100,1100)
        plt.axis('on')
        plt.show()
# Convert the list of positions into a pandas DataFrame
positions = pd.DataFrame(poss, columns=['frame', 'x', 'y','size'])

#%%
img=frames[-1]
t_lower = 50  
t_upper = 150 
edge = cv2.Canny(img, t_lower, t_upper)
cv2.imshow('original', img)
cv2.imshow('edge', edge)
cv2.waitKey(0)
cv2.destroyAllWindows()


#%% Section: Track droplets across frames

# Use trackpy to link the positions across frames into trajectories
t = tp.link_df(positions, search_range=50,
               adaptive_stop=5, adaptive_step=0.99, memory=30)

# (Optional) Filter out short-lived tracks, e.g.:
# t = tp.filter_stubs(t, 10)
trajsavedir = 'D:/Maya/Dataframes/'
trajsavename = run[:-1] + '_V3'
# Save trajectories to CSV
t.to_csv(trajsavedir + trajsavename + '.csv')

#%%Load trajectory Dataframe
folder = 'D:/Maya/Dataframes/'
filename = trajsavename + '.csv'
#filename = '250528 COM Testing 1_V1.csv'
traj_df_loaded = pd.read_csv(folder + filename,header = 0,index_col=0)


#%%Sort by particle
particles = traj_df_loaded['particle'].unique()
r_av_list = []
for p in traj_df_loaded['particle'].unique():
    print(p)
    particledf = traj_df_loaded[traj_df_loaded['particle']==p]
    r_average = np.mean(particledf['size'].to_numpy())
    r_av_list.append(r_average)
    # print(r_average)
r_av_list = np.array(r_av_list)
    
Sizedf = pd.DataFrame(data=list(zip(particles,r_av_list)),columns = ['ID','r'])
maxr=Sizedf['r'].max()

findNNdf(traj_df_loaded, 40)   
radius_map = dict(zip(particles,r_av_list)) #Based on Average from Hough Circle Fit
# radius_map = dict(zip(particles,r_centres_list))
#Update to use appropriate Radii
traj_df_loaded['size']=traj_df_loaded['particle'].map(radius_map)


frames_data = []
tol = 1.03
# for f in frametest:
for f in traj_df_loaded['frame'].unique():
    print('FRAME:' + str(f))
    framedf =   traj_df_loaded[traj_df_loaded['frame']==f]
    trajnew=findNNdf(framedf, maxr*2.5)   
    t=filter_neighbors_by_size(trajnew,tolerance=tol)
    t['frame'] = f
    
    # Append the DataFrame to the list
    frames_data.append(t)
    # print(t)
    
traj_df = pd.concat(frames_data, ignore_index=True)


#%% updated COM calculation
COM = []
traj_df['volume']=traj_df['size']**3

for f in traj_df['frame'].unique():
    frame_data = traj_df[traj_df['frame'] == f]
    total_vol = frame_data['volume'].sum()
    COMx = np.sum(frame_data.x*frame_data.volume) / total_vol
    COMy = np.sum(frame_data.y*frame_data.volume) / total_vol
    COM.append((COMx,COMy,f))
COM_df = pd.DataFrame(COM, columns=['x', 'y', 'frame'])


#%%Separation by frame
particles = traj_df['particle'].unique()
# tolerance = 0.00
frametest = [1563]
BondsList = []
SepList = []
Goodframes = []
XList = []
YList = []

# for f in frametest:
#Loop through each frame
for f in traj_df['frame'].unique():
    framedf =   traj_df[traj_df['frame']==f]
    particlesf = framedf['particle'].unique()
   
    Bonds = 0  #Variable to count num of bonds
    Mean_Sep = 0
    Count_Pairs = 0
    Good_frame = f
    CofM = Centre_of_Mass(framedf[['x','y']].to_numpy())
    XList.append(CofM[0])
    YList.append(CofM[1])

    #Loop through each unique particle pairing
    for i in  particlesf:
        for j in particlesf:
          
            if (j > i):
                particleAdf = framedf[framedf['particle']==i]
                particleBdf = framedf[framedf['particle']==j]
                #Calculate particle separation
                sep = np.sqrt((particleAdf['x'].iloc[0] - particleBdf['x'].iloc[0])**2 + 
                              (particleAdf['y'].iloc[0] - particleBdf['y'].iloc[0])**2)
                # print(sep,i,j,f)
                Mean_Sep += sep
                #If separation is less than slightly more than sum of radii, say we have a contact
                if (sep <= (Sizedf[Sizedf['ID']==i]['r'].iloc[0] + 
                    Sizedf[Sizedf['ID']==j]['r'].iloc[0]) * (tol)):
                    Contact = True
                    Bonds += 1
                else:
                    Contact = False
                # print(sep,i,j,f,Contact)
                Count_Pairs +=1
    Mean_Sep = Mean_Sep/Count_Pairs
    BondsList.append(Bonds)
    SepList.append(Mean_Sep)
    Goodframes.append(Good_frame)
        
        
        
Bondsdf = pd.DataFrame(data=list(zip(Goodframes,BondsList,SepList, XList, YList)),columns = ['frame','Num_Bonds','Mean_Sep', 'CofM_X','CofM_Y'])        
        
#%% Plot number of bonds per frame

plt.figure(figsize=(10, 6))
plt.plot(Bondsdf['frame'], Bondsdf['Num_Bonds'], marker='o',linestyle='none')
plt.title('Number of Bonds per Frame')
plt.xlabel('Frame')
plt.ylabel('Number of Bonds')
plt.grid(True)
plt.show()

#%% Distance between droplets

plt.figure(figsize=(10, 6))
plt.plot(Bondsdf['frame'], Bondsdf['Mean_Sep'], marker='o',linestyle='none')
plt.title('Distance Between Droplets')
plt.xlabel('Frame')
plt.ylabel('Distance')
plt.grid(True)
plt.show()

#%%For plotting bonds

connections=[]
norm = mpl.colors.Normalize(vmin = -10,vmax = particles[-1]+10)
cmap = plt.get_cmap('hsv')
#colorlist = ['dodgerblue','hotpink','darkorange','mediumseagreen']
# Iterate over each unique frame
for k,f in enumerate(Goodframes):
    if (k % 5 == 0):
        framedf = traj_df[traj_df['frame'] == f]
        particlesf = framedf['particle'].unique()
       
        
        # Create a new figure
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot each particle
        for j,p in enumerate(particlesf):
            print(j)
            color = cmap(norm(p))
            #color = colorlist[j]
            circle = mpl.patches.Circle((framedf[framedf['particle'] == p]['x'].iloc[0],framedf[framedf['particle'] == p]['y'].iloc[0]),
                                        Sizedf[Sizedf['ID']==p]['r'].iloc[0],facecolor=color)
            ax.add_patch(circle)
            
        # ax.scatter(framedf['x'], framedf['y'], s=framedf['size']*800, c=framedf['particle'], cmap='inferno',norm=norm,label='Parti!cles')
        X_Mid = Bondsdf[Bondsdf['frame']==f]['CofM_X'].iloc[0]
        Y_Mid = Bondsdf[Bondsdf['frame']==f]['CofM_Y'].iloc[0]
        w = 200
        plt.xlim(X_Mid-w,X_Mid+w)
        plt.ylim(Y_Mid-w,Y_Mid+w)
        # Plot lines between neighbors
        for i, row in framedf.iterrows():
            x_i, y_i = row['x'], row['y']
            for neighbor_id in row['nearest_neighbors']:
                neighbor_row = framedf[framedf['particle'] == neighbor_id].iloc[0]
                x_j, y_j = neighbor_row['x'], neighbor_row['y']
                plt.plot([x_i, x_j], [y_i, y_j], linestyle='dotted',linewidth=5, color = 'dimgrey',alpha=0.9)  # 'k-' is the color black with solid lines
    
        
        # Set plot title and labels
        ax.set_title(f"Particle Network - Frame {f}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        
        # Store the figure in the list
        connections.append(fig)
        plt.show()
        
        #savedir = 'D:/Maya/Plots/' + 'Network Plot 250520 4 Drop 2/'
        #savename = run[:-1] + '_Network_Plot_frame_' + str(f)
        #plt.savefig(savedir + savename + '.png',dpi = 200,bbox_inches = 'tight')
        
        plt.close()


#%%For plotting bonds averaged

connections=[]
norm = mpl.colors.Normalize(vmin = -10,vmax = particles[-1]+10)
cmap = plt.get_cmap('hsv')
#colorlist = ['dodgerblue','hotpink','darkorange','mediumseagreen']
# Iterate over each unique frame
for k,f in enumerate(Goodframes):
    print('Working on frame: '+str(f))
    framedf = traj_df[traj_df['frame'] == f]
    particlesf = framedf['particle'].unique()

    
    for j,p in enumerate(particlesf):
      posnew=[]
      for i in range(-2,3):
         framedfil = traj_df[traj_df['frame'] == f+i]
         filtered_row = framedfil[framedfil['particle'] == p]

         # Check if the filtered DataFrame is not empty
         if not filtered_row.empty:
             nextrow = filtered_row.iloc[0]
             x,y=nextrow['x'], nextrow['y']
             posnew.append([x,y])
             # Proceed with your logic using nextrow
               # Calculate the average position
      avg_x = sum([pos[0] for pos in posnew]) / 5
      avg_y = sum([pos[1] for pos in posnew]) / 5

      # Add the averaged position back to the original DataFrame
      traj_df.loc[(traj_df['particle'] == p) & (traj_df['frame'] == f), 'xsm'] = avg_x
      traj_df.loc[(traj_df['particle'] == p) & (traj_df['frame'] == f), 'ysm'] = avg_y  
    
    
    
for k,f in enumerate(Goodframes):
    if (k % 5 == 0):
        framedf = traj_df[traj_df['frame'] == f]
        particlesf = framedf['particle'].unique()
       
        
        # Create a new figure
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot each particle
        for j,p in enumerate(particlesf):
            print(j)
            color = cmap(norm(p))
            #color = colorlist[j]
            circle = mpl.patches.Circle((framedf[framedf['particle'] == p]['x'].iloc[0],framedf[framedf['particle'] == p]['y'].iloc[0]),
                                        Sizedf[Sizedf['ID']==p]['r'].iloc[0],facecolor=color)
            ax.add_patch(circle)
        
        # ax.scatter(framedf['x'], framedf['y'], s=framedf['size']*800, c=framedf['particle'], cmap='inferno',norm=norm,label='Parti!cles')
        X_Mid = Bondsdf[Bondsdf['frame']==f]['CofM_X'].iloc[0]
        Y_Mid = Bondsdf[Bondsdf['frame']==f]['CofM_Y'].iloc[0]
        w = 200
        plt.xlim(X_Mid-w,X_Mid+w)
        plt.ylim(Y_Mid-w,Y_Mid+w)
        # Plot lines between neighbors
        for i, row in framedf.iterrows():
            x_i, y_i = row['xsm'], row['ysm']
            for neighbor_id in row['nearest_neighbors']:
                neighbor_row = framedf[framedf['particle'] == neighbor_id].iloc[0]
                x_j, y_j = neighbor_row['xsm'], neighbor_row['ysm']
                plt.plot([x_i, x_j], [y_i, y_j], linestyle='dotted',linewidth=5, color = 'red',alpha=0.9)  # 'k-' is the color black with solid lines
                
            x_i, y_i = row['x'], row['y']
            for neighbor_id in row['nearest_neighbors']:
                neighbor_row = framedf[framedf['particle'] == neighbor_id].iloc[0]
                x_j, y_j = neighbor_row['x'], neighbor_row['y']
                plt.plot([x_i, x_j], [y_i, y_j], linestyle='dotted',linewidth=5, color = 'dimgrey',alpha=0.9)  # 'k-' is the color black with solid lines
    
        plt.show()
        # Set plot title and labels
        ax.set_title(f"Particle Network - Frame {f}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        
        # Store the figure in the list
        connections.append(fig)
        plt.show()
        
#%% Bonds plot for average vs every bond

particles = traj_df['particle'].unique()
frames_filtered = Goodframes[:-3]
BondsListNew = []
SepListNew = []

# for f in frametest:
#Loop through each frame
for f in traj_df['frame'].unique()[:-3]:
    framedf =   traj_df[traj_df['frame']==f]
    particlesf = framedf['particle'].unique()
   
    BondsNew = 0  #Variable to count num of bonds
    Mean_SepNew = 0
    Count_PairsNew = 0

    #Loop through each unique particle pairing
    for i in  particlesf:
        for j in particlesf:
          
            if (j > i):
                particleAdf = framedf[framedf['particle']==i]
                particleBdf = framedf[framedf['particle']==j]
                #Calculate particle separation
                sepNew = np.sqrt((particleAdf['xsm'].iloc[0] - particleBdf['xsm'].iloc[0])**2 + 
                              (particleAdf['ysm'].iloc[0] - particleBdf['ysm'].iloc[0])**2)
                # print(sep,i,j,f)
                Mean_SepNew += sepNew
                #If separation is less than slightly more than sum of radii, say we have a contact
                if (sepNew <= (Sizedf[Sizedf['ID']==i]['r'].iloc[0] + 
                    Sizedf[Sizedf['ID']==j]['r'].iloc[0]) * (tol)):
                    ContactNew = True
                    BondsNew += 1
                else:
                    ContactNew = False
                # print(sep,i,j,f,Contact)
                Count_PairsNew +=1
    Mean_SepNew = Mean_SepNew/Count_PairsNew
    BondsListNew.append(BondsNew)
    SepListNew.append(Mean_SepNew)
             
        
BondsdfNew = pd.DataFrame(data=list(zip(frames_filtered,BondsListNew,SepListNew)),columns = ['frame','Num_BondsNew','Mean_SepNew', 'CofM_XNew','CofM_YNew'])        



plt.figure(figsize=(10, 6))
plt.plot(Bondsdf['frame'][:-3], Bondsdf['Num_Bonds'][:-3], marker='o', linestyle='none', color='red')
plt.plot(BondsdfNew['frame'], BondsdfNew['Num_BondsNew'], marker='o',linestyle='none', color='blue')
plt.title('Number of Bonds per Frame')
plt.xlabel('Frame')
plt.ylabel('Number of Bonds')
plt.grid(True)
plt.show()



#%% Section: Plot trajectories over time

# Loop over time in steps (every 4 frames) and plot trajectory snapshots
for k in np.arange(framei,framef-1, 4):
    print(k)  # Print current frame number
    plotTraj(traj_df_loaded, k, directory, frames, COM_df)  # Plot trajectory on this frame
    
        
#%% COM distance between frames

# Extract the initial COM coordinates
initial_COMx=COM_df.loc[COM_df['frame'] == framei, 'x'].values[0]
initial_COMy=COM_df.loc[COM_df['frame'] == framei, 'y'].values[0]

# Calculate the Euclidean distance for each frame
COM_df['distance'] = np.sqrt((COM_df['x'] - initial_COMx)**2 + (COM_df['y'] - initial_COMy)**2)


plt.figure(figsize=(10, 6))
plt.plot(COM_df['frame'], COM_df['distance'], marker='o', linestyle='none', color='hotpink')
plt.title('Distance of COM from Initial Position Over Time')
plt.xlabel('Frame')
plt.ylabel('Distance (units)')
plt.grid(True)
plt.show()

#%% Using B/W to tell if two points are in contact


def count_white_black_pixels_on_line(img, pt1, pt2, white_threshold=200):
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.line(mask, (x1, y1), (x2, y2), 255, 1)
    line_pixels = img[mask == 255]
    white_count = np.sum(line_pixels >= white_threshold)
    black_count = np.sum(line_pixels < white_threshold)
    return white_count, black_count
'''
# --- Define the endpoints of the line you want to track ---
pt1 = (725, 540)
pt2 = (850, 560)

# --- Loop through all frames in Goodframes ---
for f in Goodframes:
    index = Goodframes.index(f)
    img = frames[index]  # Grayscale image for this frame

    white, black = count_white_black_pixels_on_line(img, pt1, pt2)

    print(f"Frame {f}: Line from {pt1} to {pt2}")
    print(f"  White pixels: {white}")
    print(f"  Black pixels: {black}")

    if white <= 5:
        print("These droplets are likely TOUCHING.\n")
    else:
        print("These droplets are likely NOT touching.\n")
   '''     
    
for k,f in enumerate(Goodframes):
    if (k % 5 == 0):
        framedf = traj_df[traj_df['frame'] == f]
        particlesf = framedf['particle'].unique()
        index = Goodframes.index(f)
        img = frames[index]
        
        
        plt.xlim(X_Mid-w,X_Mid+w)
        plt.ylim(Y_Mid-w,Y_Mid+w)
        # Plot lines between neighbors
        for i, row in framedf.iterrows():
            x_i, y_i = row['xsm'], row['ysm']
     
            for neighbor_id in row['nearest_neighbors']:
                neighbor_row = framedf[framedf['particle'] == neighbor_id].iloc[0]
                x_j, y_j = neighbor_row['xsm'], neighbor_row['ysm']
                pt1=(x_i,y_i)
                pt2=(x_j,y_j)
                white, black = count_white_black_pixels_on_line(img, pt1, pt2)
                print('frame: '+str(f)+', white: '+str(white)+ ', black: '+str(black))
                
                

