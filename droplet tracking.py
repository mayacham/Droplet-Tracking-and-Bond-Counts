# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 12:28:52 2025

@author: Kari
"""

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

##need to fix COM because it is using unweighted at one point
## fix neighbour
## theres a section to filter out short lived trajectories - use that for the random circles that appear so we can se bounds lower

#THIS IS THE WORKING CODE EDIT STUFF HERE NOT IN THE ONE WITH THE TWO


import numpy as np #numerical opperations
import matplotlib.pyplot as plt #plotting graphs and images
import os #file and directory operations
from skimage.filters import unsharp_mask #to sharpen images
from skimage.feature import match_template, peak_local_max #for template matching and peak detection
from skimage.color import rgba2rgb, rgb2gray  # for image color conversions to grayscale
# Data handling and tracking libraries
import pandas as pd               # For dataframes (tables of data)
import pims                       # For loading image sequences (videos)
import trackpy as tp              # For tracking particles across frames
import imageio                    # For reading and writing image files
import matplotlib as mpl          # For advanced plotting (colormaps, etc.)
import cv2                        # OpenCV library for additional image processing
from sklearn.neighbors import KDTree  # For efficient nearest neighbour searches

# Define an image preprocessing function using PIMS pipeline
# Preprocess each image frame: crop, convert to grayscale, sharpen, and normalize
@pims.pipeline
def preprocess_img(frame):
    # Crop the image to focus on the region of interest and avoid processing unnecessary parts ---------------------------------
    frame = frame[0:1300, 0:1600]
    
    #the next few lines are to convert a RGBA image into a grayscale image
    # If the image has 4 channels (RGBA), convert it to RGB
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = rgba2rgb(frame)
    # If the image has 3 channels (RGB), convert it to grayscale
    if frame.ndim == 3 and frame.shape[2] == 3:
        frame = rgb2gray(frame)
    # Convert image values from float (0-1) to 8-bit integers (0-255)
    frame = (frame * 255).astype(np.uint8)
    frame = unsharp_mask(frame, radius=2, amount=5)     # Sharpen the image using unsharp masking to enhance droplet features
    frame *= 255.0 / frame.max()     # Normalize the sharpened image back to 8-bit range from 0-255
    return frame.astype(np.uint8)     # Return the final preprocessed image   


def plotTrajold(traj, k, directory, frames, framei):
    fig = plt.figure(figsize=(12, 8), dpi=150)  # Create a new figure
    plt.ylim(600, 800)
    plt.xlim(700, 1000)
    plt.title(f"Trajectories up to Frame {k}")
    subset = traj[traj.frame <= k]                  # Filter the trajectory DataFrame to include only particles from frame 0 up to and including frame `k`
    tp.plot_traj(subset, colorby='particle', cmap=mpl.cm.winter,
                 superimpose=frames[k-framei], plot_style={'linewidth': 2}) # Plot the trajectories using trackpy, overlaying them on the corresponding image frame
    return fig      # Return the figure object so it can be saved or closed outside this function


# Function to plot trajectories up to a specific frame number `k`
def plotTraj(traj, k, directory, frames,COM):
    plt.figure(figsize=(12, 8), dpi=150)  #     # Create a new figure with specified size and resolution
    plt.clf()  # Clear current figure (in case it’s being reused)

    subset = traj[traj.frame <= k]     # Filter the trajectory DataFrame to include only frames up to `k`
    # Set the limits of the x and y axes to zoom in on region of interest --------------------------------------
    plt.xlim(300,1200)
    plt.ylim(50,1000)
    # Plot the center of mass at the current frame 'k'  as a red X
    plt.scatter(COM_df.loc[COM_df['frame'] == k, 'x'].values[0], COM_df.loc[COM_df['frame'] == k, 'y'].values[0],
                color='red', marker='x', s=100, label='Center of Mass')
    # Plot the center of mass at the initial frame 'framei' as a blue X
    plt.scatter(COM_df.loc[COM_df['frame'] == framei, 'x'].values[0], COM_df.loc[COM_df['frame'] == framei, 'y'].values[0], 
                color='dodgerblue', marker='x', s=75, label='Initial Center of Mass')
    # Plot trajectories using trackpy, coloring by particle ID, overlaid on the video frame at frame 'k'
    tp.plot_traj(subset, colorby='particle', cmap=mpl.cm.winter,
                 superimpose=frames[k-framei], plot_style={'linewidth': 2})
    
    
def findareas(trajectories):
    plt.figure(figsize=(8, 8))
    unique_particles = trajectories['particle'].unique()
    areas=[]
    for particle_id in unique_particles:
        part = trajectories[trajectories['particle'] == particle_id]
        # Skip if too few points to form a loop
        if len(part) < 3:
            continue
        x = part['x'].values
        y = part['y'].values
        # Close the loop
        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])
        # Shoelace formula
        area = 0.5 * np.abs(np.dot(x_closed[:-1], y_closed[1:]) - np.dot(x_closed[1:], y_closed[:-1]))
        areas.append(area)
        # Plot the loop
        plt.fill(x_closed, y_closed, alpha=0.3, label=f'Particle {particle_id}, Area={area:.2f}')
        plt.plot(x_closed, y_closed, marker='o', linestyle='-')
    plt.title('Closed Loops for All Particles')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #plt.tight_layout()
    plt.show()
    return areas
    

# Function to find nearest neighbours using KDTree for fast and effiecient spatial search
# adds columns to the dataframe indicating each particle's neighbours and the number of neighbours.
'''
Parameters:
df : pandas.DataFrame
    DataFrame containing at least 'x', 'y', and 'particle' columns.
rad : float
    The radius within which neighbours are searched for. Use rad > droplet size to account for position errors.
Returns:
df : pandas.DataFrame
    The input dataframe with two new columns: 
    'nearest_neighbours' (list of neighboring particles)
    'num_neighbours' (number of neighbours found)
'''
def findNNdf(df, rad): 
    # Extract x and y coordinates from the DataFrame
    points = df[['x', 'y']].values     
    # Build the KDTree with the given points for fast neighbour searching
    tree = KDTree(points, leaf_size=2)    
    # Find the nearest neighbours within the given radius using the tree
    nn_indices = tree.query_radius(points, r=rad)   
    # Map the indices to particle numbers using the 'particle' column
    particle_numbers = df['particle'].values
    nn_particle_numbers = [particle_numbers[i] for i in nn_indices]
    # Count the number of neighbours for each particle (excluding itself)
    numNN = np.array([len(i) - 1 for i in nn_indices])  
    # Add the nearest neighbour data to the original DataFrame
    df['nearest_neighbors'] = nn_particle_numbers
    df.loc[:,'num_neighbors'] = numNN
    #return df, nn_fraction_df
    return df


# Function to filter neighbours based on particle sizes and exclude neighbours too far apart
def filter_neighbors_by_size(info,tolerance=1.1):
    '''
    This function filters neighbours based on size and distance.
    A neighbour is only kept if its distance is smaller than the sum of both particle sizes * tolerance factor.
    Parameters:
    info : pandas.DataFrame
        DataFrame containing 'x', 'y', 'size', 'particle', and 'nearest_neighbours' columns.
    tolerance : float
        Multiplier to slightly increase neighbour distance allowance (default: 1.1).
    Returns:
    df : pandas.DataFrame
        Updated dataframe with filtered neighbours and updated 'num_neighbours' count.
    '''
    df=info #  renaming for local clarity
    # Get the particle positions and sizes
    particles = df['particle'].values
    sizes = df['size'].values
    x_coords = df['x'].values
    y_coords = df['y'].values
    # Lists to store updated neighbour lists and neighbour counts when looping
    updated_neigh_list = []
    numneigh=[]
    
    #loop over each particle
    for i, particle_id in enumerate(particles):
        # Reset index to avoid accidental misalignment
        df = df.reset_index(drop=True)
        # Retrieve the current list of neighbours IDs for this particle
        neighbors = df[df['particle']==particle_id]['nearest_neighbors'].iloc[0]
        # Get the current particle's position and size
        x_i, y_i, size_i = x_coords[i], y_coords[i], sizes[i]
        # Start an empty list for valid neighbours
        valid_neighbors = []
        # Loop through neighbours and check distance condition
        for neighbor_id in neighbors:
            if neighbor_id == particle_id:     #This will skip calling itself it's own neighbour
                continue
            # Find neighbor's index in dataframe
            neighbor_idx = df[df['particle'] == neighbor_id].index[0]
            # Get the neighbor's position and size
            x_j, y_j, size_j = x_coords[neighbor_idx], y_coords[neighbor_idx], sizes[neighbor_idx]
            # Calculate Euclidean distance between current particle and neighbor
            distance = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
            # Check if the distance is less than or equal to the sum of their radius times the tolerance
            if distance <= (size_i + size_j)*tolerance:
                valid_neighbors.append(neighbor_id) #keep neighbour if fits requirement
        # Store valid neighbours for current particle
        updated_neigh_list.append(valid_neighbors)
        numneigh.append(len(valid_neighbors))
        
    # Replace the 'nearest_neighbours' column with the filtered list of valid neighbours
    df['nearest_neighbors'] = updated_neigh_list
    df.loc[:,'num_neighbors'] = numneigh
    return df


#calculate COM from a set of all particle positions
def Centre(coords):
    totalx = np.mean(coords[:,0])
    totaly = np.mean(coords[:,1])
    return np.array([totalx,totaly])

#%% Loading images and setup

# Define file paths to locate the image sequence
directory = 'D:/Maya/'                   # Main directory where your data is stored
run = '250711 Testing Droplet Tracking 4/'     # Subfolder for this specific experiment-------------------------
prefix = '*.tiff'  # File pattern to load TIFF images

# Define frame range to analyze (can adjust as needed - must be at least filtering length (30))-----------------------------------------------------------
framei = 7000                                # Starting frame index
framef = 9700                              # Ending frame index

version = "_V4 (7000-9700)"                  #for later when saving dataframe --------------------------------------------

# Load image sequence using PIMS
sequence = pims.ImageSequence(os.path.join(directory + run + prefix))
# Preprocess the loaded image frames using your preprocessing pipeline. Only frames between framei and framef are preprocessed
frames = preprocess_img(sequence[framei:framef])

# Set up directory and filename for saving the trajectories and other plots
trajsavedir = 'D:/Maya/Dataframes/'      # Folder to save output CSV files
plotsavedir = 'D:/Maya/Plots/'         # Main folder for all plots 
savename = run[:-1] + version          # Generate filename based on run name and version

filesavedir = os.path.join(plotsavedir, savename)  # Creates a path for your run's folder
os.makedirs(filesavedir, exist_ok=True)             # Creates a folder needed to later save images

#%% Loop through all frames and detect peaks

poss = []       # Initialize an empty list to store detected circle positions for each frame
# Loop over each frame (except the last one)
for i in range(framei,framef-1):
    img=frames[i-framei]     # Get the preprocessed image for the current frame
    blur = cv2.GaussianBlur(img, (5, 5), 0)  # Optional Gaussian blur - can be used to smooth image before detection
    #blur=img     # Currently no additional blurring is applied
    inv = cv2.bitwise_not(blur)     # Invert image colors if needed (useful if droplets are dark on a light background)
    # Apply Hough Circle Transform to detect circular droplets
    circles = cv2.HoughCircles(
        blur,                   # Input image (must be 8-bit grayscale)
        cv2.HOUGH_GRADIENT,     # Detection method (gradient-based)
        dp=1.2,                 # Inverse ratio of accumulator resolution to image resolution
        minDist=20,             # Minimum distance between detected circle centers
        param1=180,             # High threshold for Canny edge detector (edge detection step)
        param2=27,              # Accumulator threshold for circle detection (lower = more circles detected, until about 30 where it starts to make up circles)
        minRadius=10,           # Minimum expected circle radius
        maxRadius=60            # Maximum expected circle radius
    )
    # Draw circles
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)     # Convert image to BGR so circles can be drawn in color for visualization
    # If any circles were detected in this frame
    if circles is not None:
        circles = np.uint16(np.around(circles[0])) # Round circle parameters to integers
        # Loop through all detected circles
        for x, y, r in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)       # Draw detected circle outline (green)
            cv2.circle(output, (x, y), 2, (255, 255, 255), 3)     # Draw detected circle center (white dot (tiny circle))
            poss.append([i, x, y, r])             # Store frame number, center coordinates, and radius into list

 # Occasionally print frame progress every x frames
    #if (i%100==0):        
    if (i%1==0):        
        print('Frame = ' + str(i))
    # Show result
        plt.imshow(output)
        plt.title(f"Hough Circle Detection - Frame {i}")
        # x and y axis of graph ------------------------------------------------------------
        plt.xlim(200,1550) 
        plt.ylim(0,1250) 
        plt.axis('on')
        plt.show()
# After processing all frames, convert the collected positions into a pandas DataFrame
positions = pd.DataFrame(poss, columns=['frame', 'x', 'y','size'])



# Use trackpy to link detected droplet positions across frames into trajectories
t = tp.link_df(
    positions,      # DataFrame of detected droplet positions (from Hough transform)
    search_range=100,    # Maximum distance a droplet can move between frames (in pixels)
    adaptive_stop=50,    # Allows dynamic adjustment of search range after 5 frames
    adaptive_step=0.99, # Adjustment factor for adaptive search range
    memory=10           # Number of frames a droplet can disappear and still be linked
) 

'''  
###POTENTIAL OPTION IF IMPROVED TO TRACK USING SIZE INSTEAD OR COMBINE TWO
scale_factor = 200 / 10  # = 20
positions['size_scaled'] = positions['size'] * scale_factor

t = tp.link_df(
    positions,
    search_range=50,
    pos_columns=['x', 'y', 'size_scaled'],
    memory=5)
'''

# Optional: Filter out short-lived trajectories that exist for fewer than 30 frames
t = tp.filter_stubs(t, 30)   #uncomment this line if you want to apply filtering-----------------------

# plot the centres of each circle to ensure all are identified in the first frame
img = frames[0]      # Select the first frame image
centers = t[t['frame'] == framei]       # Select the corresponding centers (particles) for the first frame
# Plot the image and centers
plt.figure()
plt.imshow(img, cmap='gray', zorder=0)
plt.scatter(centers['x'], centers['y'], c='r', s=40, edgecolors='white', linewidth=0.5, zorder=1)
plt.title(f'Frame {framei}')
plt.axis('off')
plt.tight_layout()
plt.show()
plt.clf()

# checking which particles are lost track of and new IDs are assigned
total_frames = t['frame'].nunique()  # Count of unique frames
for pid in t['particle'].unique():      # Loop over each unique particle ID
    group = t[t['particle'] == pid]    # Select rows belonging to this particle id
    n = len(group)  # Number of frames where this particle appears
    # Check if it matches the total frames
    if n != total_frames:
        print(f"Particle {pid} appears in {n}/{total_frames} frames — missing {total_frames - n}")
    else:
        print(f"Particle {pid} appears in all {total_frames} frames")

# plot trajectories of droplets to make sure none are given new IDs 
for k in np.arange(framei, framef-1, 1):
    fig = plotTrajold(t, k, directory, frames, framei)  # Get the figure from the function
    fig.savefig(os.path.join(filesavedir, savename + f'_frame_{k}.png'), dpi=300, bbox_inches='tight')  # Save using fig
    plt.close(fig)

# Save the full trajectory dataframe to CSV for later analysis
t.to_csv(trajsavedir + savename + '.csv')
# Load saves trajectory DataFrame from csv into a pandas dataframe
traj_df_loaded = pd.read_csv(trajsavedir + savename + '.csv', header = 0, index_col = 0)


# Area calculations

areas=findareas(t)          #runs the function find areas

averageArea = np.mean(areas)        #calculate the average area for each of the areas in the areas list
print (averageArea)












#%% Sort by particle 

particles = traj_df_loaded['particle'].unique() # Get a list of unique particle IDs from the trajectory DataFrame
r_av_list = [] # Initialize a list to store the average radius for each particle

# Loop over each unique particle
for p in traj_df_loaded['particle'].unique():
    #print(p)  # Print particle ID (for tracking progress/debugging)
    particledf = traj_df_loaded[traj_df_loaded['particle'] == p]     # Filter the DataFrame for only the current particle
    # Calculate the average radius (size) of that particle over all frames
    r_average = np.mean(particledf['size'].to_numpy())
    r_av_list.append(r_average)
    #print(r_average)  # Print r average (for tracking progress/debugging)
# Convert the list of average radii into a NumPy array
r_av_list = np.array(r_av_list)

Sizedf = pd.DataFrame(data=list(zip(particles, r_av_list)), columns=['ID', 'r']) # Create a new DataFrame that maps particle ID to average radius
maxr = Sizedf['r'].max() # Find the largest average radius (used for neighbor search later)

# Run nearest-neighbor detection once on the full DataFrame.This adds 'nearest_neighbors' and 'num_neighbors' columns to traj_df_loaded,
findNNdf(traj_df_loaded, 40) 

radius_map = dict(zip(particles, r_av_list)) # Create a dictionary that maps each particle ID to its average radius, pairing the two
traj_df_loaded['size'] = traj_df_loaded['particle'].map(radius_map) # Update the 'size' column in the original DataFrame to use this average radius instead of per-frame values

frames_data = [] # Initialize a list to store neighbor-filtered data for each frame
tol = 1.03   # Set tolerance for neighbor size comparison (used to filter close-enough neighbors)

# Loop through each frame in the dataset
for f in traj_df_loaded['frame'].unique():
    #print('FRAME:' + str(f))  # Print frame number for tracking progress
    framedf = traj_df_loaded[traj_df_loaded['frame'] == f]     # Filter the data for just this frame
    trajnew = findNNdf(framedf, maxr * 2.5)     # Find initial nearest neighbors using KDTree with a radius based on max particle size
    t = filter_neighbors_by_size(trajnew, tolerance=tol)     # Filter neighbors based on proximity and particle size similarity
    t['frame'] = f     # Add frame number back to the filtered DataFrame (since it gets detached in processing)
    
    frames_data.append(t)     # Add this frame's processed data to the list
# Combine all frame-level DataFrames into a single full trajectory DataFrame
traj_df = pd.concat(frames_data, ignore_index=True)


#%% COM calculation

# Initialize an empty list to store COM values for each frame
COM = []
traj_df['volume']=traj_df['size']**3 # Calculate the volume of each droplet by cubing its size (4/3 cancels out later)

# Loop through each unique frame in the trajectory DataFrame
for f in traj_df['frame'].unique():
    frame_data = traj_df[traj_df['frame'] == f]     # Extract all particles in the current frame
    total_vol = frame_data['volume'].sum()     # Calculate the total volume of all particles in this frame
    # Compute the volume-weighted average x and y coordinates (COM position)
    COMx = np.sum(frame_data.x*frame_data.volume) / total_vol
    COMy = np.sum(frame_data.y*frame_data.volume) / total_vol
    
    # Store the COM coordinates and corresponding frame number in a list
    COM.append((COMx,COMy,f))
# Convert the list of COM values into a pandas DataFrame with labeled columns
COM_df = pd.DataFrame(COM, columns=['x', 'y', 'frame'])


#%%Separation by frame

particles = traj_df['particle'].unique() # Get all unique particle IDs in the dataset

# Initialize lists to store data for each frame
BondsList = []   # Number of particle-particle bonds per frame
SepList = []     # Mean separation per frame
Goodframes = []  # Frames successfully analyzed
XList = []       # Center of mass x-coordinate
YList = []       # Center of mass y-coordinate

#Loop through each frame in the dataset
for f in traj_df['frame'].unique():
    # Filter data for the current frame
    framedf =   traj_df[traj_df['frame']==f]
    particlesf = framedf['particle'].unique()
   
    # Initialize metrics for this frame
    Bonds = 0  #Variable to count num of bonds
    Mean_Sep = 0
    Count_Pairs = 0
    Good_frame = f
       
    # Calculate geometric centre
    Cen = Centre(framedf[['x','y']].to_numpy())
    XList.append(Cen[0])
    YList.append(Cen[1])
    

    #Loop through each unique particle pairing
    for i in  particlesf:
        for j in particlesf: #avoid double counting
            if (j > i):
                # Extract position info for particle i and j
                particleAdf = framedf[framedf['particle']==i]
                particleBdf = framedf[framedf['particle']==j]
                #Calculate particle separation
                sep = np.sqrt((particleAdf['x'].iloc[0] - particleBdf['x'].iloc[0])**2 + 
                              (particleAdf['y'].iloc[0] - particleBdf['y'].iloc[0])**2)
                Mean_Sep += sep  # Accumulate separation for averaging
                
                # Check if particles are in contact (within scaled sum of radii)
                if (sep <= (Sizedf[Sizedf['ID']==i]['r'].iloc[0] + 
                    Sizedf[Sizedf['ID']==j]['r'].iloc[0]) * (tol)):
                    Contact = True
                    Bonds += 1  # Count as a bond if close enough
                else:
                    Contact = False
                Count_Pairs += 1  # Keep track of how many pairs checked

    Mean_Sep = Mean_Sep/Count_Pairs        # Average separation across all pairs in this frame
    # Record metrics for the frame
    BondsList.append(Bonds)
    SepList.append(Mean_Sep)
    Goodframes.append(Good_frame)
# Store all collected metrics into a DataFrame for further analysis and plotting        
Bondsdf = pd.DataFrame(data=list(zip(Goodframes,BondsList,SepList, XList, YList)),columns = ['frame','Num_Bonds','Mean_Sep', 'Centre_X','Centre_Y'])        

#%% Plot number of bonds per frame

plt.figure(figsize=(10, 6)) # Create a new figure with a specific size
plt.plot(Bondsdf['frame'], Bondsdf['Num_Bonds'], marker='o', linestyle='none') # Plot the number of bonds in each frame (as individual points, no connecting lines)
plt.title('Number of Bonds per Frame')
plt.xlabel('Frame')
plt.ylabel('Number of Bonds')
plt.grid(True)
plt.show() # Display the plot

#%% Plot distance between droplets

plt.figure(figsize=(10, 6)) # Create a new figure with a specific size
plt.plot(Bondsdf['frame'], Bondsdf['Mean_Sep'], marker='o', linestyle='none') # Plot the mean separation distance between particles per frame (individual points)
plt.title('Distance Between Droplets')
plt.xlabel('Frame')
plt.ylabel('Distance')
plt.grid(True)
plt.show() # Display the plot

#%% Plotting droplets and bonds

connections=[]  # List to store the figure objects for each plotted frame
# Set up color normalization and colormap for visualizing different particles
norm = mpl.colors.Normalize(vmin=particles.min(), vmax=particles.max()) # Create a normalization object that maps particle IDs (from min to max) onto a [0, 1] scale
cmap = plt.get_cmap('hsv') # Choose a colormap ('hsv') that spans the full hue spectrum, giving a rainbow-like set of colors
#colorlist = ['dodgerblue','hotpink','darkorange','mediumseagreen'] #use if only working with a small amount of droplets to assign colours

# Iterate over each unique frame 
for k,f in enumerate(Goodframes):
    if (k % 5 == 0):  # Only plot every 5th frame to reduce clutter
        framedf = traj_df[traj_df['frame'] == f] # Get particle data for the current frame
        particlesf = framedf['particle'].unique() # List of unique particle IDs in this frame
       
        fig, ax = plt.subplots(figsize=(6, 6))        # Create a new figure for plotting

        # Plot each particle as a circle
        for j,p in enumerate(particlesf):
            color = cmap(norm(p)) # Assign a unique color based on particle ID
            #color = colorlist[j] # use if working with colorlist not cmap
            # Create a matplotlib circle patch at the particle's position with its radius
            circle = mpl.patches.Circle(
                (framedf[framedf['particle'] == p]['x'].iloc[0],   # X-coordinate of particle p
                 framedf[framedf['particle'] == p]['y'].iloc[0]),  # Y-coordinate of particle p
                Sizedf[Sizedf['ID'] == p]['r'].iloc[0],            # Radius of particle p
                facecolor=color                                    # Fill color based on particle ID
                  )
            ax.add_patch(circle)      # Add this particle circle to the current plot
              
        # Get center of mass for this frame and set plot limits to center around it*************************************************
        X_Mid = Bondsdf[Bondsdf['frame']==f]['Centre_X'].iloc[0] # Get the X-coordinate of geometrical centre for the current frame from `Bondsdf`
        Y_Mid = Bondsdf[Bondsdf['frame']==f]['Centre_Y'].iloc[0] # Get the Y-coordinate of geometrical centre for the current frame
        w = 600  # Half-width/height of the viewing window
        plt.xlim(X_Mid-w,X_Mid+w) # Set X-axis limits to zoom in on region centered around the geometrical centre
        plt.ylim(Y_Mid-w,Y_Mid+w) # Set Y-axis limits similarly
        
        # Draw dotted lines between each particle and its nearest neighbuors
        for i, row in framedf.iterrows():
            x_i, y_i = row['x'], row['y']  # Get the (x, y) position of this particle
            for neighbor_id in row['nearest_neighbors']:     # For each of its nearest neighbours...
                neighbor_row = framedf[framedf['particle'] == neighbor_id].iloc[0]
                x_j, y_j = neighbor_row['x'], neighbor_row['y']         # Get the (x, y) position of the neighbour
                # Draw a dotted line between the particle and its neighbour
                plt.plot([x_i, x_j], [y_i, y_j],
                         linestyle='dotted',
                         linewidth=5,
                         color='dimgrey',
                         alpha=0.9)
        # Set plot title and labels
        ax.set_title(f"Particle Network - Frame {f}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        
        # Store the figure in the list
        connections.append(fig)
        plt.show()  # Display the current plot
        plt.close()  # Close the figure to free up memory

#%% Plotting droplets and bonds averaged

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
        X_Mid = Bondsdf[Bondsdf['frame']==f]['Centre_X'].iloc[0]
        Y_Mid = Bondsdf[Bondsdf['frame']==f]['Centre_Y'].iloc[0]
        w = 400
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
  
 
#%% Combined version
# Optional: Smooth positions across time for averaged plotting
for f in Goodframes:
    framedf = traj_df[traj_df['frame'] == f]
    for p in framedf['particle'].unique():
        posnew = []
        for i in range(-2, 3):
            tempdf = traj_df[traj_df['frame'] == f + i]
            row = tempdf[tempdf['particle'] == p]
            if not row.empty:
                posnew.append(row[['x', 'y']].iloc[0].tolist())
        if len(posnew) == 5:
            avg_x = sum(pos[0] for pos in posnew) / 5
            avg_y = sum(pos[1] for pos in posnew) / 5
            traj_df.loc[(traj_df['particle'] == p) & (traj_df['frame'] == f), ['xsm', 'ysm']] = [avg_x, avg_y]

# General plotting function
def plot_frame(f, use_smoothed=False, window_size=600):
    framedf = traj_df[traj_df['frame'] == f]
    particlesf = framedf['particle'].unique()
    fig, ax = plt.subplots(figsize=(6, 6))

    for j, p in enumerate(particlesf):
        color = cmap(norm(p))
        particle_row = framedf[framedf['particle'] == p]

        # Use smoothed coords if requested and available, else raw coords
        if use_smoothed and 'xsm' in particle_row.columns and not pd.isna(particle_row['xsm'].iloc[0]):
            x = particle_row['xsm'].iloc[0]
            y = particle_row['ysm'].iloc[0]
        else:
            x = particle_row['x'].iloc[0]
            y = particle_row['y'].iloc[0]

        r = Sizedf[Sizedf['ID'] == p]['r'].iloc[0]
        circle = mpl.patches.Circle((x, y), r, facecolor=color)
        ax.add_patch(circle)

    # Draw neighbor connections
    for _, row in framedf.iterrows():
        for neighbor_id in row['nearest_neighbors']:
            
            neighbor_row = framedf[framedf['particle'] == neighbor_id].iloc[0]
            x_coords = [row['x'], neighbor_row['x']]
            y_coords = [row['y'], neighbor_row['y']]
            line_color = 'dimgrey'

            # Choose which coords to plot for neighbors
            if use_smoothed and ('xsm' in row and 'xsm' in neighbor_row) and \
               (not pd.isna(row['xsm']) and not pd.isna(neighbor_row['xsm'])):
                x_coords = [row['xsm'], neighbor_row['xsm']]
                y_coords = [row['ysm'], neighbor_row['ysm']]
                line_color = 'red'
               
            ax.plot(x_coords, y_coords, linestyle='dotted', linewidth=5, color=line_color, alpha=0.9)

    # Center the plot
    X_Mid = Bondsdf[Bondsdf['frame'] == f]['Centre_X'].iloc[0]
    Y_Mid = Bondsdf[Bondsdf['frame'] == f]['Centre_Y'].iloc[0]
    ax.set_xlim(X_Mid - window_size, X_Mid + window_size)
    ax.set_ylim(Y_Mid - window_size, Y_Mid + window_size)

    ax.set_title(f"Particle Network - Frame {f} {'(Smoothed)' if use_smoothed else '(Raw)'}")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    plt.show()
    plt.close()
#### problem when a dot is not picked up for the average plot - makes it a point 0,0 which throws off the rest of the averages and gives a random dot on the graph
for f in Goodframes[2:-2]:  # Ensure full 5-frame window exists
    if f % 5 == 0:
        #plot_frame(f, use_smoothed=False)
        plot_frame(f, use_smoothed=True)

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
                
