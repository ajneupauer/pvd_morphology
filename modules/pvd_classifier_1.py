# Import modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
from skimage import io, morphology, filters
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from skan import csr
import seaborn as sns
import cv2
from collections import Counter
from scipy.interpolate import splprep, splev

"""Create a class for the neurite random forest classifier model"""
class PVDNeuriteClassifier:
    """
    Initialize as a blank random forest classifier with set number of estimators and method of class balancing.
    """
    def __init__(self, estimators = 100, class_weight = 'balanced'):
        self.model = RandomForestClassifier(
            n_estimators=estimators, 
            random_state=42
            )
        self.graph = None
        self.skeleton = None
        self.branch_data = None
        self.image = None
    
    """
    Load a trained model.
    """
    def load_model(self, model_path: str):
        self.model = joblib.load(model_path)
    
    """
    Load and preprocess fluorescence image.
    Also skeletonize the mask and return the skeleton. 
    """
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        # If it's already binary, use as is; otherwise threshold
        if img.dtype == bool or np.all(np.isin(img, [0, 1])):
            binary = img > 0
        else:
            # Apply Gaussian smoothing first
            img_smooth = filters.gaussian(img, sigma=0.5)
            # Use Otsu thresholding
            threshold = filters.threshold_otsu(img_smooth)
            binary = img_smooth > threshold
        
        # Clean up small artifacts
        binary = morphology.remove_small_objects(binary, min_size=150)
        
        # Add vertical/horizontal filter layers to binary image
        img = img.astype(np.uint8)
        # Morphological opening with vertical structuring element (filter emphasizes vertical elements)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        vertical_branches = cv2.morphologyEx(img, cv2.MORPH_OPEN, vertical_kernel)
        # Morphological opening with horizontal structuring element (filter emphasizes horizontal elements) 
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        horizontal_branches = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel)
        # Reconstruct to preserve connections
        vertical_reconstructed = cv2.morphologyEx(vertical_branches, cv2.MORPH_DILATE, 
                                                cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))
        vertical_final = cv2.bitwise_and(vertical_reconstructed, img)
        horizontal_reconstructed = cv2.morphologyEx(horizontal_branches, cv2.MORPH_DILATE, 
                                                cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))
        horizontal_final = cv2.bitwise_and(horizontal_reconstructed, img)
        # Remove overlap between vertical and horizontal
        vertical_only = cv2.subtract(vertical_final, horizontal_final)
        horizontal_only = cv2.subtract(horizontal_final, vertical_final)
        
        # Add classifier attribute image, 
        # a three-layer image with the original mask and horizontal/vertical filters
        height, width = img.shape
        self.image = np.empty([3, height, width], dtype = np.uint8)
        self.image[0] = binary
        self.image[1] = vertical_only
        self.image[2] = horizontal_only
        
        # Skeletonize mask
        pre_skeleton = morphology.skeletonize(binary)
        # Remove terminal branch artifacts (less than 10 px long)
        skeleton = prune_terminal_branches(pre_skeleton, min_length=10)
        
        self.skeleton = skeleton
        
        return skeleton
    
    """
    Convert skeleton to graph representation.
    Each node is a pixel in the skeleton, and each edge represents connections between adjacent pixels.
    """
    def extract_graph(self, skeleton: np.ndarray) -> nx.Graph:
        # Find intersection points and endpoints
        branch_points = self._find_branch_points(skeleton)
        end_points = self._find_end_points(skeleton)
        
        # Create a graph with nodes at intersection points and endpoints
        G = nx.Graph()
        
        # Add all points as nodes
        for point in np.argwhere(skeleton):
            G.add_node(tuple(point), pos=tuple(point), type='segment')
        
        # Mark intersection points and endpoints
        for point in branch_points:
            G.nodes[tuple(point)]['type'] = 'branch'
        for point in end_points:
            G.nodes[tuple(point)]['type'] = 'end'
        
        # Connect adjacent pixels in the skeleton
        for i, j in np.argwhere(skeleton):
            if skeleton[i, j]: # If the given point isn't background
                # Check all 8 neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        # Skip the self point
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj # Current neighbor point
                        # If the neighbor point exists and is not background, add edge between self and neighbor
                        if 0 <= ni < skeleton.shape[0] and 0 <= nj < skeleton.shape[1] and skeleton[ni, nj]:
                            G.add_edge((i, j), (ni, nj))
        
        self.graph = G
        return G
    
    """
    Find intersection points in the skeleton.
    """
    def _find_branch_points(self, skeleton: np.ndarray) -> np.ndarray:
        # Define the kernel for convolution
        kernel = np.array([
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ])
        # Convolve the skeleton with the kernel
        conv = ndimage.convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
        # Intersection points have values >= 13 (center + 3 or more neighbors)
        branch_points = np.argwhere(conv >= 13)
        
        return branch_points
    
    """
    Find endpoints in the skeleton.
    """
    def _find_end_points(self, skeleton: np.ndarray) -> np.ndarray:
        # Define the kernel for convolution
        kernel = np.array([
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ])
        # Convolve the skeleton with the kernel
        conv = ndimage.convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
        # Endpoints have values = 11 (center + 1 neighbor)
        end_points = np.argwhere(conv == 11)
        
        return end_points
    
    """
    Separate the skeleton graph into branch segments.
    Each branch is an entry in a DataFrame providing its points and basic stats.
    """
    def segment_neurites(self) -> pd.DataFrame:
        if self.graph is None:
            raise ValueError("Graph not created. Run extract_graph first.")
        
        # Get list of only graph nodes representing intersection and endpoints
        all_neighbors = self.graph.adj
        b_and_e_pts = [n for n, attr in self.graph.nodes(data=True) if attr['type'] != 'segment']
        
        segments = []
        visited = [] # Track nodes already visited
        
        # Find paths connecting each intersection/endpt to all its adjacent intersection/endpts
        for pt in b_and_e_pts:
            b_pt_neighbors = list(all_neighbors[pt])
            # Trace a path towards each neighbor
            for neighbor in b_pt_neighbors:
                if neighbor not in visited: # If the neighbor is already visited, this path has already been found
                    # Initialize a path from the current pt towards one of its neighbors 
                    segment = [pt, neighbor]
                    # Keep extending the path until the growing end encounters 'branch' or 'end' nodes
                    while self.graph.nodes[segment[-1]]['type'] == 'segment':
                        seg_neighbors = list(all_neighbors[segment[-1]])
                        nodes_added = 0
                        # Look at all neighbors of the growing end
                        for i in range(len(seg_neighbors)):
                            # Only add neighbors not already in the segment
                            # The penulimate pt in the path will be a neighbor of the growing end, but we don't want to add it again
                            if seg_neighbors[i] != segment[-2 - nodes_added]:
                                segment.append(seg_neighbors[i])
                                nodes_added += 1
                    # Only add paths longer than 5 pixels to the list of segments
                    if len(segment) > 5:            
                        segments.append(segment)
                        # The penultimate pt in the path is the neighbor of another branch/endpt
                        # Prevents from double-counting reciprocal paths (e.g.: A -> B, B -> A)
                        visited.append(segment[-2]) 
        
        # Basic features for each segment
        segment_features = []
        n = 0
        for i, segment in enumerate(segments):
            num_pts = len(segment)
            if num_pts > 5: # Again, ensure only paths longer than 5 pixels are included
                length, orientation, curvature, tortuosity, waviness = branch_geom(segment)
                segment_features.append({
                    'id': n,
                    'length': length,
                    'orientation': orientation,
                    'curvature': curvature,
                    'tortuosity': tortuosity,
                    'waviness': waviness,
                    'segment': segment
                })
                n += 1
        
        self.branch_data = pd.DataFrame(segment_features)
        return self.branch_data
    
    """
    Extract features from segments for machine learning.
    """
    def extract_features(self, img: np.ndarray, max_proj: np.ndarray) -> pd.DataFrame:
        # Preprocess image
        skeleton = self.preprocess_image(img)
        # Extract graph
        self.extract_graph(skeleton)
        # Convert to segments
        segments = self.segment_neurites()
        
        # Extract additional features for classification and add to basic features
        features = []
        for idx, row in segments.iterrows():
            segment = row['segment']
            # Get x and y coords at each point
            x_pos = [pt[1] for pt in segment]
            y_pos = [pt[0] for pt in segment]
            
            # Average intensity
            # !!! Consider image normalization
            intensities = [max_proj[y_pos[i], x_pos[i]] for i in range(len(segment))]
            avg_intensity = np.mean(intensities)
            
            # Segment position
            midpt_x = np.mean(x_pos)
            rel_x = midpt_x / skeleton.shape[1]
            midpt_y = np.mean(y_pos)
            
            # Local density (how many skeleton pixels in neighborhood of the midpt)
            neighborhood_size = 200 # Makes 401 x 401 px box
            y_min = round(max(0, midpt_y - neighborhood_size))
            y_max = round(min(skeleton.shape[0], midpt_y + neighborhood_size))
            x_min = round(max(0, midpt_x - neighborhood_size))
            x_max = round(min(skeleton.shape[1], midpt_x + neighborhood_size))
            local_density = np.sum(skeleton[y_min:y_max, x_min:x_max]) / ((y_max-y_min) * (x_max-x_min))
            
            # Get horizontalness and verticalness
            hCount = 0 # Track horizontal and total points encountered
            ptCount = 0
           
            for point in segment:
                # If the point is in the vertical filter...
                if self.image[1, point[0], point[1]] == 1:
                    ptCount += 1
                # If the point is in the horizontal filter, add to horizontal point count
                if self.image[2, point[0], point[1]] == 1:
                    hCount += 1
                    ptCount += 1
            # Horizontalness = proportion of points in the horizontal filter
            if ptCount != 0: 
                hNess = hCount / ptCount
            else: 
                hNess = 0
            
            features.append({
                'id': row['id'],
                'length': row['length'],
                'orientation': row['orientation'],
                'curvature': row['curvature'],
                'tortuosity': row['tortuosity'],
                'waviness': row['waviness'],
                'horizontal_likely': hNess > 0.5,
                'relative_y': midpt_y / skeleton.shape[0],
                'relative_x': rel_x,
                # Applies heavy weight to relative x pos where quaternaries are common
                'quat_filter': 0.2 * np.sin(3 * np.pi * rel_x) + 0.5,
                'local_density': local_density,
                'average_intensity': avg_intensity
            })
        
        return pd.DataFrame(features)
    
    """
    Train the model with labeled data.
        Data: neurite masks with their associated MIPs
        Labels: colormaps indicating what branch label exists at each pixel
    """
    def train(self, images: list[np.ndarray], color_maps: list[np.ndarray], max_projs: list[np.ndarray], balance_method='class_weight'):
        # Part 1: Extract features from all images
        all_features = []
        all_labels = []
        
        for i in range(len(images)):
            # Extract features
            features = self.extract_features(images[i], max_projs[i])
            # Get labels
            label_path = color_maps[i]
            image_labels = get_labels(label_path, self.branch_data)
            # Add labels
            for idx, row in features.iterrows():
                segment_id = row['id']
                # Only proceed if the segment has a label and it is not zero
                if segment_id in image_labels and image_labels[segment_id] != 0:
                    # Add image's labels and features
                    all_features.append(row.drop(['id', 'length']).values) # Don't need length or id
                    all_labels.append(image_labels[segment_id])
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Part 2: Check class distribution
        class_counts = Counter(y)
        print("Class distribution:")
        for class_label, count in sorted(class_counts.items()):
            print(f"Class {class_label}: {count} samples")
        
        # Part 3: Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Part 4: Evaluate model
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        # Feature importance: how useful is the feature to classification?
        feature_names = [
            'orientation',
            'curvature',
            'tortuosity',
            'waviness',
            'horizontal_likely',
            'relative_y',
            'relative_x',
            'quat_filter',
            'local_density',
            'average_intensity'
        ]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(importance_df)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    """
    Predict dendrite types in a new image.
    """    
    def predict(self, img: np.ndarray, max_proj: np.ndarray) -> pd.DataFrame:
        # Extract features
        features = self.extract_features(img, max_proj)
        # Predict
        predictions = self.model.predict(features.drop(['id', 'length'], axis=1).values)
        # Add predictions to features
        features['dendrite_type'] = predictions
        
        return features
    
    """
    Visualize the results of the model classifying a neurite mask.
    Specify an image, which will set the base layer of the visual.
    """
    def visualize(self, image_path: str, predictions=None):# -> matplotlib.figure.Figure:
        
        # Load base image for visualization
        image = io.imread(image_path)
        
        # If predictions not provided, make them
        if predictions is None:
            predictions = self.predict(image_path)
        # Remove branches without a prediction assigned
        predictions = predictions[predictions['dendrite_type'] != 0]
        
        # Create color mapping for dendrite types
        color_map = {
            1: '#1f77b4', # 1° dendrites, blue
            2: '#2ca02c', # 2° dendrites, green
            3: '#ffbb33', # 3° dendrites, yellow
            4: '#d62728', # 4° dendrites, red
            5: '#6400cf', # artifacts, purple
            6: '#ff00ff', # !!! axon coming soon 
        }
        
        # Create a visualization image
        fig, ax = plt.subplots(figsize=(10, 100))
        ax.imshow(image, cmap = 'gray')
        
        # Draw each segment with its predicted type/color
        for idx, row in predictions.iterrows():
            # !!! previously: segment = self.branch_data.loc[idx, 'segment']
            segment = row['segment']
            dendrite_type = row['dendrite_type']
            # Convert segment to array for plotting
            segment_arr = np.array(segment)
            ax.plot(segment_arr[:, 1], segment_arr[:, 0], color=color_map[dendrite_type], linewidth=6)
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return fig

"""Helper functions"""

"""
Given a set of branches in a DataFrame, get the labels of the branches at their midpoints.
Creates dictionary lookup structure where:
    Key: index of the branch
    Value: label of the branch midpoint
"""
def get_labels(labeled_image: np.ndarray, branch_data: pd.DataFrame) -> dict:
    labels = {}
    # Go through all branches
    for n in range(len(branch_data)):
        branch = branch_data.loc[n]['segment']
        midpt = branch[len(branch) // 2]
        # Key n paired with value of the labeled_image/colormap at the branch midpt
        labels[n] = int(labeled_image[midpt])
    
    return labels

"""
Given a skeleton extracted from a neurite mask, remove artifactual branches.
Sometimes, thick neurites create small spurs (short terminal branches) where there is no intersection.
Branches must be longer than min_length or they are pruned.
"""
def prune_terminal_branches(skeleton: np.ndarray, min_length=5) -> np.ndarray:
    # Get data on skeleton branches
    skel = csr.Skeleton(skeleton)
    summary = csr.summarize(skel, separator = '-')
    pruned_skeleton = skeleton.copy()

    # Look for branches to remove
    for i, row in summary.iterrows():
        # Only prune if the branch connects to exactly one junction (i.e., it's a spur)
        # These are terminal branches: one side has no junction
        branch_type = row['branch-type']
        is_short = row['branch-distance'] < min_length

        # If the branch is short and terminal (type 1), set its points to False in the skeleton
        if branch_type == 1 and is_short:
            coords = skel.path_coordinates(i)[1:-1]
            for y, x in coords:
                pruned_skeleton[int(round(y)), int(round(x))] = False
        
        # If the branch is short and and isolated branch (type 0) or loop (type 3), remove it
        # Allow a less stringent length cutoff, permiting longer branches
        if branch_type == 0 or branch_type == 3:
            if row['branch-distance'] < min_length * 3:
                coords = skel.path_coordinates(i)[1:-1]
                for y, x in coords:
                    pruned_skeleton[int(round(y)), int(round(x))] = False
        
    return pruned_skeleton

"""
Compute curvature and orientation of a branch using spline fitting.
"""
def orientation_curvature_via_spline(points: np.ndarray, s=0.1) -> tuple: # Use m - np.sqrt(2*m), m = # points
    # Get spline
    tck, u = splprep(points.T, s=s, k=3)
    # Evaluate derivatives
    velocity = np.array(splev(u, tck, der=1)).T
    acceleration = np.array(splev(u, tck, der=2)).T
    # Compute parametric curvature at each point
    cross = velocity[:, 0] * acceleration[:, 1] - velocity[:, 1] * acceleration[:, 0]
    curvature = cross / np.linalg.norm(velocity, axis=1)**3
    # Compute orientations from tangent vectors at each point
    angles = np.arctan2(velocity[:, 1], velocity[:, 0])
    
    return angles, curvature

"""
Compute the length, average orientation, and three metrics on degree of meandering of a branch.
    Length: each left/right up/down movement = 1, each diagonal movement = sqrt(2)
    Average orientation: the orientation of the branch, on average
    Curvature: average unsigned parametric curvature along the branch
    Tortuosity: ratio of branch euclidean length to path length
    Waviness: number of sign changes in parametric curvature, normalized by length
"""
def branch_geom(branch: list[tuple]) -> tuple:
    branch = np.array(branch)
    
    # Compute length
    length = 0

    # Change in x and y at each point
    # array([[delx1, delx2, ..., delxn], 
    #       [dely1, dely2, ..., delyn]])
    diffs = np.abs(np.diff(branch, axis=0))
    
    for i in range(diffs.shape[0]):
        point = diffs[i, :]
        # Apply dist formula for each step and add to total length
        px_dist = np.sqrt(point[0]**2 + point[1]**2)
        length += px_dist
    
    length = round(length)

    # Euclidean length and tortuosity
    euclidean_length = np.linalg.norm(np.array([branch[-1, 0] - branch[0, 0], branch[-1, 1] - branch[0, 1]]))
    tortuosity = length / euclidean_length
    
    # Parametric curvature and waviness
    s = branch.shape[0] - np.sqrt(2 * branch.shape[0]) # Recommended to use m - np.sqrt(2*m), m = # points
    angles, curvature = orientation_curvature_via_spline(branch, s)
    sign_changes = np.sum(np.diff(np.sign(curvature)) != 0)
    waviness = sign_changes / length
    mean_curve = np.mean(np.abs(curvature))
    
    # Handle angle wrapping for orientation mean
    avg_orientation = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    avg_orientation = np.degrees(avg_orientation)
    
    return length, avg_orientation, mean_curve, tortuosity, waviness