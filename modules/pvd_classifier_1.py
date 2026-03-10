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
#from imblearn.over_sampling import SMOTE

class PVDNeuriteClassifier:
    def __init__(self, estimators = 100, class_weight = 'balanced'):
        self.model = RandomForestClassifier(
            n_estimators=estimators, 
            random_state=42
            )
        self.graph = None
        self.skeleton = None
        self.branch_data = None
        self.image = None
    
    def load_model(self, model_path):
        self.model = joblib.load(model_path)
    
    def preprocess_image(self, img):
        """Load and preprocess fluorescence image"""
        
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
        # Morphological opening with vertical structuring element
        img = img.astype(np.uint8)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        vertical_branches = cv2.morphologyEx(img, cv2.MORPH_OPEN, vertical_kernel)

        # Morphological opening with horizontal structuring element  
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
        
        height, width = img.shape
        
        self.image = np.empty([3, height, width], dtype = np.uint8)
        self.image[0] = binary
        self.image[1] = vertical_only
        self.image[2] = horizontal_only
        
        pre_skeleton = morphology.skeletonize(binary)
        skeleton = prune_terminal_branches(pre_skeleton, min_length=10)
        
        self.skeleton = skeleton
        
        return skeleton
    
    def extract_graph(self, skeleton):
        """Convert skeleton to graph representation"""
        # Find branch points and endpoints
        branch_points = self._find_branch_points(skeleton)
        end_points = self._find_end_points(skeleton)
        
        # Create a graph with nodes at branch points and endpoints
        G = nx.Graph()
        
        # Add all points as nodes
        for point in np.argwhere(skeleton):
            G.add_node(tuple(point), pos=tuple(point), type='segment')
        
        # Mark branch points and endpoints
        for point in branch_points:
            G.nodes[tuple(point)]['type'] = 'branch'
        
        for point in end_points:
            G.nodes[tuple(point)]['type'] = 'end'
        
        # Connect adjacent pixels in the skeleton
        for i, j in np.argwhere(skeleton):
            if skeleton[i, j]:
                # Check 8 neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < skeleton.shape[0] and 0 <= nj < skeleton.shape[1] and skeleton[ni, nj]:
                            G.add_edge((i, j), (ni, nj))
        
        self.graph = G
        return G
    
    def _find_branch_points(self, skeleton):
        """Find branch points in the skeleton"""
        # Define the kernel for convolution
        kernel = np.array([
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ])
        
        # Convolve the skeleton with the kernel
        conv = ndimage.convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
        
        # Branch points have values >= 13 (center + 3 or more neighbors)
        branch_points = np.argwhere(conv >= 13)
        
        return branch_points
    
    def _find_end_points(self, skeleton):
        """Find end points in the skeleton"""
        # Define the kernel for convolution
        kernel = np.array([
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ])
        
        # Convolve the skeleton with the kernel
        conv = ndimage.convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
        
        # End points have values = 11 (center + 1 neighbor)
        end_points = np.argwhere(conv == 11)
        
        return end_points
    
    def segment_neurites(self):
        """Segment the neuron into branches and classify them"""
        if self.graph is None:
            raise ValueError("Graph not created. Run extract_graph first.")
        
        all_neighbors = self.graph.adj
        b_and_e_pts = [n for n, attr in self.graph.nodes(data=True) if attr['type'] != 'segment']
        
        segments = []
        visited = [] # track nodes already visited
        
        # Find paths connecting each branch/end pt to all its adjacent branch/end pts
        for pt in b_and_e_pts:
            b_pt_neighbors = list(all_neighbors[pt])
            for neighbor in b_pt_neighbors:
                if neighbor not in visited: # if the neighbor is already visited, this path already exists
                    # Initialize a path from the current pt towards one of its neighbors 
                    segment = [pt, neighbor]
                    # Keep extending the path until 'branch' or 'end' nodes are encountered
                    while self.graph.nodes[segment[-1]]['type'] == 'segment':
                        seg_neighbors = list(all_neighbors[segment[-1]])
                        nodes_added = 0
                        for i in range(len(seg_neighbors)):
                            # Only add neighbors not already in the segment
                            if seg_neighbors[i] != segment[-2 - nodes_added]:
                                segment.append(seg_neighbors[i])
                                nodes_added += 1
                    if len(segment) > 5:            
                        segments.append(segment)
                        visited.append(segment[-2]) # the penultimate pt in the path is the neighbor of another branch/end pt
        
        # Extract features for each segment
        segment_features = []
        n = 0
        for i, segment in enumerate(segments):
            num_pts = len(segment)
            if num_pts > 5:
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
    
    def extract_features(self, img, max_proj):
        """Extract features for machine learning"""
        # Preprocess image
        skeleton = self.preprocess_image(img)
        
        # Extract graph
        self.extract_graph(skeleton)
        
        # Segment neurites
        segments = self.segment_neurites()
        
        # Extract additional features for classification
        features = []
        for idx, row in segments.iterrows():
            segment = row['segment']
            # Get x and y coords at each point
            x_pos = [pt[1] for pt in segment]
            y_pos = [pt[0] for pt in segment]
            
            # Average intensity
            #intensities = []
            #for i in range(len(segment)):
            #    if x_pos[i] < skeleton.shape[1] and y_pos[i] < skeleton.shape[0]:
            #        intensities.append(max_proj[x_pos[i], y_pos[i]])
            intensities = [max_proj[y_pos[i], x_pos[i]] for i in range(len(segment))]
            avg_intensity = np.mean(intensities)
            
            # Segment position
            midpt_x = np.mean(x_pos)
            rel_x = midpt_x / skeleton.shape[1]
            midpt_y = np.mean(y_pos)
            
            # Local density (how many skeleton pixels in neighborhood)
            neighborhood_size = 200
            y_min = round(max(0, midpt_y - neighborhood_size))
            y_max = round(min(skeleton.shape[0], midpt_y + neighborhood_size))
            x_min = round(max(0, midpt_x - neighborhood_size))
            x_max = round(min(skeleton.shape[1], midpt_x + neighborhood_size))
            local_density = np.sum(skeleton[y_min:y_max, x_min:x_max]) / ((y_max-y_min) * (x_max-x_min))
            
            # Get horizontalness and verticalness
            hCount = 0
            ptCount = 0
           
            for point in segment:
                if self.image[1, point[0], point[1]] == 1:
                    ptCount += 1
                if self.image[2, point[0], point[1]] == 1:
                    hCount += 1
                    ptCount += 1
            
            if ptCount != 0: 
                hNess = hCount / ptCount
            else:
                hNess = 0
            
            features.append({
                'id': row['id'],
                'length': row['length'],
                'orientation': row['orientation'],
                'curvature': row['curvature'],
                #'tortuosity': row['tortuosity'],
                'waviness': row['waviness'],
                'horizontal_likely': hNess > 0.5,
                'relative_y': midpt_y / skeleton.shape[0],
                'relative_x': rel_x,
                'quat_filter': 0.2 * np.sin(3 * np.pi * rel_x) + 0.5,
                #'quat_filter': 338 * ((rel_x - 0.5) ** 6) - 180 * ((rel_x - 0.5) ** 4) + 24 * ((rel_x - 0.5) ** 2),
                #'tert_filter': np.e ** ((-0.5 * (rel_x - 0.25)/0.1) ** 2) + np.e ** ((-0.5 * (rel_x - 0.75)/0.1) ** 2),
                #'sec_filter': np.e ** ((-0.5 * (rel_x - 0.35)/0.08) ** 2) + np.e ** ((-0.5 * (rel_x - 0.65)/0.08) ** 2),
                'local_density': local_density,
                'average_intensity': avg_intensity
            })
        
        return pd.DataFrame(features)
    
    def train(self, images, color_maps, max_projs, balance_method='class_weight'):
        """Train the model with labeled data"""
        # Extract features from all images
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
                if segment_id in image_labels and image_labels[segment_id] != 0:
                    all_features.append(row.drop(['id', 'length']).values)
                    all_labels.append(image_labels[segment_id])
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Check class distribution
        class_counts = Counter(y)
        print("Class distribution:")
        for class_label, count in sorted(class_counts.items()):
            print(f"Class {class_label}: {count} samples")
        
        # Apply balancing
        #if balance_method == 'smote':
        #    smote = SMOTE(random_state=42)
        #    X, y = smote.fit_resample(X, y)
        
        #    class_counts = Counter(y)
        #    print("Class distribution after balancing:")
        #    for class_label, count in sorted(class_counts.items()):
        #        print(f"Class {class_label}: {count} samples")
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_names = [
            'orientation',
            'curvature',
            #'tortuosity',
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
        
    def predict(self, img, max_proj):
        """Predict dendrite types in a new image"""
        # Load image
        #image = io.imread(image_path)
        
        # Extract features
        features = self.extract_features(img, max_proj)
        
        # Predict
        predictions = self.model.predict(features.drop(['id', 'length'], axis=1).values)
        
        # Add predictions to features
        features['dendrite_type'] = predictions
        
        return features
    
    def visualize(self, image_path, predictions=None):
        """Visualize the classification results"""
        # Load image
        image = io.imread(image_path)
        
        # If predictions not provided, make them
        if predictions is None:
            predictions = self.predict(image_path)
        
        predictions = predictions[predictions['dendrite_type'] != 0]
        
        # Create a color map for dendrite types
        
        #color_map = {
        #    1: '#1f77b4',    # 1° dendrites, blue
        #    2: '#ff7f0e',   # 2° dendrites, orange
        #    3: '#2ca02c',  # 3° dendrites, green
        #    4: '#d62728',   # 4° dendrites, red
        #    5: '#ffbb33',
        #    6: '#ff00ff' 
        #}
        
        color_map = {
            1: '#1f77b4', # 1° dendrites, blue
            2: '#2ca02c', # 2° dendrites, green
            3: '#ffbb33', # 3° dendrites, yellow
            4: '#d62728', # 4° dendrites, red
            5: '#6400cf', # artifacts, purple
            6: '#ff00ff' 
        }
        
        # Create a visualization image
        fig, ax = plt.subplots(figsize=(10, 100))
        ax.imshow(image, cmap = 'gray')
        
        # Draw each segment with its predicted type
        for idx, row in predictions.iterrows():
            segment = self.branch_data.loc[idx, 'segment']
            dendrite_type = row['dendrite_type']
            
            # Convert segment to array for plotting
            segment_arr = np.array(segment)
            ax.plot(segment_arr[:, 1], segment_arr[:, 0], color=color_map[dendrite_type], linewidth=6)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='1° dendrite'),
            Line2D([0], [0], color='green', lw=2, label='2° dendrite'),
            Line2D([0], [0], color='yellow', lw=2, label='3° dendrite'),
            Line2D([0], [0], color='red', lw=2, label='4° dendrite')
        ]
        #ax.legend(handles=legend_elements)
        
        #plt.title('PVD Neuron Dendrite Classification')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # Generate statistics
        #dendrite_counts = predictions['dendrite_type'].value_counts().sort_index()
        #print("Dendrite Type Counts:")
        #for dendrite_type, count in dendrite_counts.items():
        #    print(f"{dendrite_type}° dendrites: {count}")
        
        # Calculate total length per dendrite type
        #dendrite_lengths = {}
        #for dendrite_type in range(1, 5):
        #    total_length = predictions[predictions['dendrite_type'] == dendrite_type]['length'].sum()
        #    dendrite_lengths[dendrite_type] = total_length
        
        #print("\nDendrite Type Total Lengths:")
        #for dendrite_type, length in dendrite_lengths.items():
        #    print(f"{dendrite_type}° dendrites: {length} pixels")
        
        return fig

def get_labels(labeled_image, branch_data):
    labels = {}
    
    for n in range(len(branch_data)):
        branch = branch_data.loc[n]['segment']
        midpt = branch[len(branch) // 2]
        labels[n] = int(labeled_image[midpt])
    
    return labels

def prune_terminal_branches(skeleton, min_length=5):
    skel = csr.Skeleton(skeleton)
    summary = csr.summarize(skel, separator = '-')
    #paths = skel.path_coordinates()

    pruned_skeleton = skeleton.copy()

    for i, row in summary.iterrows():
        # Only prune if the branch connects to exactly one junction (i.e., it's a spur)
        # These are endpoint branches: one side has no junction
        branch_type = row['branch-type']
        is_short = row['branch-distance'] < min_length

        if branch_type == 1 and is_short:
            coords = skel.path_coordinates(i)[1:-1]
            for y, x in coords:
                pruned_skeleton[int(round(y)), int(round(x))] = False
        
        if branch_type == 0 or branch_type == 3:
            if row['branch-distance'] < min_length * 3:
                coords = skel.path_coordinates(i)[1:-1]
                for y, x in coords:
                    pruned_skeleton[int(round(y)), int(round(x))] = False
        
    return pruned_skeleton

def orientation_curvature_via_spline(points, s=0.1): # Use m - np.sqrt(2*m), m = # points
    """Compute curvature using spline fitting"""
    tck, u = splprep(points.T, s=s, k=3)
    
    # Evaluate derivatives
    velocity = np.array(splev(u, tck, der=1)).T
    acceleration = np.array(splev(u, tck, der=2)).T
    
    # Compute curvature
    cross = velocity[:, 0] * acceleration[:, 1] - velocity[:, 1] * acceleration[:, 0]
    curvature = cross / np.linalg.norm(velocity, axis=1)**3
    
    # For 2D: compute angles from tangent vectors
    angles = np.arctan2(velocity[:, 1], velocity[:, 0])
    
    return angles, curvature

def branch_geom(branch):
    branch = np.array(branch)
    
    length = 0
    diffs = np.abs(np.diff(branch, axis=0))
    
    for i in range(diffs.shape[0]):
        point = diffs[i, :]
        px_dist = np.sqrt(point[0]**2 + point[1]**2)
        length += px_dist
    
    length = round(length)
    euclidean_length = np.linalg.norm(np.array([branch[-1, 0] - branch[0, 0], branch[-1, 1] - branch[0, 1]]))
    tortuosity = length / euclidean_length
    
    s = branch.shape[0] - np.sqrt(2 * branch.shape[0])
    angles, curvature = orientation_curvature_via_spline(branch, s)
    sign_changes = np.sum(np.diff(np.sign(curvature)) != 0)
    waviness = sign_changes / length
    mean_curve = np.mean(np.abs(curvature))
    
    # Handle angle wrapping for circular mean
    avg_orientation = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    avg_orientation = np.degrees(avg_orientation)
    
    return length, avg_orientation, mean_curve, tortuosity, waviness

