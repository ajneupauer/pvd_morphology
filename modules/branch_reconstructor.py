#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 11:33:11 2025

@author: alexneupauer
"""

from collections import defaultdict

def connect_segments(segments: list, threshold=5.0, max_step_ratio=3.0) -> list:
    """
    Connect segments with nearby endpoints, with validation to prevent aberrant lines.
    
    Args:
        segments: List of segments, each as [(x1,y1), (x2,y2), ...]
        threshold: Maximum distance to consider endpoints "near"
        max_step_ratio: Maximum ratio of connection step to average segment step
    
    Returns:
        List of connected segments
    """
    
    if not segments:
        return []
    
    # Build connection graph
    connections = build_connection_graph(segments, threshold, max_step_ratio)
    
    # Find connected components and build chains
    used = set()
    merged_segments = []
    
    for start_seg in range(len(segments)):
        if start_seg in used:
            continue
            
        # Build the longest possible chain starting from this segment
        chain = build_optimal_chain(segments, connections, start_seg, used)
        if chain and len(chain) >= 10:
            merged_segments.append(chain)
    
    return merged_segments


"""
Build a graph of valid connections between segments.
The graph is a list of connections, where each connection is a dictionary with four keys:
    'target': id of the branch that the source branch connects to
    'type': of the four possible ways to connect these segments, which is it?
    'distance': gap distance between the two segments if connected
    'needs_reverse': would the target segment need its points order reversed if connecting
    it to the source segment?
"""
def build_connection_graph(segments: list, threshold: float, max_step_ratio: float) -> dict[int, list]:    
    connections = defaultdict(list)
    
    # Check all pairs of segments i and j where i < j
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            # Check all possible ways these segments could connect
            valid_connections = find_valid_connections(
                segments[i], segments[j], i, j, threshold, max_step_ratio
            )
            # Add each valid connection and its reverse
            for conn in valid_connections:
                connections[i].append(conn)
                # Add reverse connection !!!
                reverse_conn = reverse_connection(conn, source_idx = i)
                connections[j].append(reverse_conn)
    
    return connections

"""
Find all valid ways two segments can connect.
Each pair of segments can connect end to start, end to end, start to start, or start to end.
But not every option makes sense (is valid).
A valid connection maintains a small distance between the end/start of separate segments:
    The gap between segments should not exceed a user-defined threshold distance.
    The gap should also be limited based on the average step size of the segments
    so we don't bridge a big gap between two segments whose points are sparsely sampled 
    just because it's under the flat threshold. This limit is determined by a user-defined
    ratio which multiplies by the average step size.
"""
def find_valid_connections(
        seg1: list, seg2: list, idx1: int, idx2: int, threshold: float, max_step_ratio: float
        ) -> list[dict]:
    
    valid_connections = []
    
    # Get endpoints
    start1, end1 = seg1[0], seg1[-1]
    start2, end2 = seg2[0], seg2[-1]
    
    # Calculate average step sizes for validation
    avg_step1 = calculate_average_step(seg1)
    avg_step2 = calculate_average_step(seg2)
    # We don't want to take a step between segments much larger than interpoint distances
    max_allowed_step = max(avg_step1, avg_step2) * max_step_ratio
    
    # Check all four possible endpoint connections
    connections_to_check = [
        (end1, start2, idx2, 'end_to_start', False),    # seg1.end -> seg2.start
        (end1, end2, idx2, 'end_to_end', True),         # seg1.end -> seg2.end (reverse seg2)
        (start1, start2, idx2, 'start_to_start', True), # seg1.start -> seg2.start (reverse seg2, prepend)
        (start1, end2, idx2, 'start_to_end', False),    # seg1.start -> seg2.end (prepend)
    ]
    
    for p1, p2, target_idx, conn_type, needs_reverse in connections_to_check:
        dist = distance(p1, p2)
        # A connection between two segments is valid if the gap is not too big
        # i.e. cannot exceed threshold distance of maximum allowed step
        if dist <= threshold and dist <= max_allowed_step:
            valid_connections.append({
                'target': target_idx,
                'type': conn_type,
                'distance': dist,
                'needs_reverse': needs_reverse
            })
    
    return valid_connections

"""
Create the reverse connection for bidirectional graph.
"""
def reverse_connection(conn: dict, source_idx: int) -> dict:
    # Maps each connection type to its reverse connection type
    type_map = {
        'end_to_start': 'start_to_end',
        'end_to_end': 'end_to_end',
        'start_to_start': 'start_to_start',
        'start_to_end': 'end_to_start'
    }
    
    # Return the same dictionary, except the type is reversed
    return {
        'target': source_idx,
        'type': type_map.get(conn['type'], conn['type']),
        'distance': conn['distance'],
        'needs_reverse': conn['needs_reverse']
    }

"""
Build the optimal chain starting from a start_seg.
"""
def build_optimal_chain(segments: list, connections: dict[int, list], start_seg: int, used: set) -> list:
    
    if start_seg in used:
        return None
    
    # Try building chains in different ways and pick the longest valid one
    best_chain = None
    best_length = 0
    best_used = set()
    
    # Try extending forward from end
    chain, chain_used = extend_chain(segments, connections, start_seg, used, 'forward')
    if len(chain) > best_length:
        best_chain, best_length, best_used = chain, len(chain), chain_used
    
    # Try extending backward from start  
    chain, chain_used = extend_chain(segments, connections, start_seg, used, 'backward')
    if len(chain) > best_length:
        best_chain, best_length, best_used = chain, len(chain), chain_used
    
    # Try extending in both directions
    chain, chain_used = extend_chain_bidirectional(segments, connections, start_seg, used)
    if len(chain) > best_length:
        best_chain, best_length, best_used = chain, len(chain), chain_used
    
    # Mark all segments in winning strategy as globally used
    used.update(best_used)
    
    # If extension fails, just return the start segment
    return best_chain if best_chain else list(segments[start_seg])

"""
Extend chain in one direction. Begins at a starting segment and keeps finding 
connections to the growing end in the specified direction. At each step, the best
(smallest gap) connection is chosen and chain extension is validated. The loop stops
when no more connections can be found or extension fails.
"""
def extend_chain(
        segments: list, connections: dict[int, list], start_seg: int, global_used: set, direction: str
        ) -> tuple[list, set]:
    
    chain = list(segments[start_seg])
    used = {start_seg}
    current_seg = start_seg
    
    # Loop to look for valid connections from the most recently added segment and apply them
    while True:
        best_connection = None
        best_distance = float('inf')
        
        # Find the best valid connection
        for conn in connections[current_seg]: # Get all segments connecting to the current one
            target_seg = conn['target']
            
            # Skip target segment if its already been used in a branch reconstruction
            if target_seg in used or target_seg in global_used:
                continue
            
            # Skip target segment if its connection type is invalid for the supplied direction
            if not is_valid_direction(conn['type'], direction):
                continue
            
            # If we encounter a target with a smaller gap than the current one,
            # update the best connection
            if conn['distance'] < best_distance:
                best_connection = conn
                best_distance = conn['distance']
        
        # If no valid connection was found, stop extending the chain
        if not best_connection:
            break
        
        # Apply the connection (actually join the target to the growing chain)
        target_seg = best_connection['target']
        new_chain = apply_connection(chain, segments[target_seg], best_connection, direction)
        
        # Ensure the chain was extended properly
        if new_chain and is_valid_chain_extension(chain, new_chain):
            chain = new_chain # update growing chain
            used.add(target_seg) # update used segments
            current_seg = target_seg # update current segment
        # If it was not extended properly, stop extending
        else:
            break
    
    return chain, used

"""
Extend chain in both directions. Begins at a starting segment to extend_chain()
forward. Then extends a backwards chain from the starting segment. Finally, 
add the forward and backward chains together.
"""
def extend_chain_bidirectional(
        segments: list, connections: dict[int, list], start_seg: int, global_used: set
        ) -> tuple[list, set]:
    
    chain = list(segments[start_seg])
    used = {start_seg}
    
    # Extend forward from the original segment
    forward_chain, forward_used = extend_chain(segments, connections, start_seg, global_used, 'forward')
    
    # Extend backward from the original segment
    backward_chain, backward_used = extend_chain(segments, connections, start_seg, global_used | forward_used, 'backward')
    
    # Check if forward and backward extensions were successful
    forward_extension_failed = len(forward_chain) <= len(segments[start_seg])
    backward_extension_failed = len(backward_chain) <= len(segments[start_seg])
    
    # No successful extension in either direction -> return starting segment
    if forward_extension_failed and backward_extension_failed:
        return chain, used
    # No successful forward extension -> return backward_chain
    elif forward_extension_failed:
        combined_chain = backward_chain
    # No successful backward extension -> return forward_chain
    elif backward_extension_failed:
        combined_chain = forward_chain
    # Successful extension in both directions ->
    # Combine: backward_chain + forward_chain[1:] (skip duplicate middle segment)
    else:
        combined_chain = backward_chain + forward_chain[len(segments[start_seg]):]
    
    return combined_chain, forward_used | backward_used

"""
Check if connection type is valid for the given direction.
"""
def is_valid_direction(conn_type: str, direction: str) -> bool:
    # If we begin connection from a source's end, we are going forward
    if direction == 'forward':
        return conn_type in ['end_to_start', 'end_to_end']
    # If we begin connection from a source's start, we are going backward
    elif direction == 'backward':
        return conn_type in ['start_to_start', 'start_to_end']
    
    return True

"""
Apply a connection between current chain and next segment.
"""
def apply_connection(current_chain: list, next_segment: list, connection: dict, direction: str) -> list:
    # Try attaching the next_segment
    try:
        conn_type = connection['type']
        needs_reverse = connection.get('needs_reverse', False)
        
        # Prepare the segment to add
        # Reverse segment if the connection type requires it
        segment_to_add = list(reversed(next_segment)) if needs_reverse else list(next_segment)
        
        if direction == 'forward':
            if conn_type == 'end_to_start':
                # current_chain.end connects to segment_to_add.start
                return current_chain + segment_to_add[1:]  # Skip duplicate point
            elif conn_type == 'end_to_end':
                # current_chain.end connects to segment_to_add.end (already reversed if needed)
                return current_chain + segment_to_add[1:]
        
        elif direction == 'backward':
            if conn_type == 'start_to_end':
                # segment_to_add.end connects to current_chain.start
                return segment_to_add[:-1] + current_chain  # Skip duplicate point
            elif conn_type == 'start_to_start':
                # segment_to_add.start connects to current_chain.start (already reversed if needed)
                return segment_to_add[:-1] + current_chain
    
    # If connecting the next_segment fails, return None
    except (IndexError, TypeError):
        return None
    
    return None

"""
Validate that the chain extension doesn't create problems. Examines the splice
junction to see if points around it contain consecutive duplicate points or sharp
direction changes. If either of these occur, the extension is invalid.
"""
def is_valid_chain_extension(old_chain: list, new_chain: list) -> bool:
    # If the new chain is not longer than the old one, extension is invalid
    if len(new_chain) <= len(old_chain):
        return False
    
    # Check for potential errors
    if len(new_chain) >= 3:
        # Look at the connection point area ([-2, +2) from the splice point)
        connection_area = new_chain[len(old_chain)-2:len(old_chain)+2]
        if len(connection_area) >= 3:
            # Check for duplicate consecutive points !!!
            for i in range(len(connection_area) - 1):
                if connection_area[i] == connection_area[i + 1]:
                    return False
            # Check for sharp reversals that might indicate errors
            for i in range(len(connection_area) - 2):
                # Look for reversals in consecutive point windows of three
                p1, p2, p3 = connection_area[i], connection_area[i+1], connection_area[i+2]
                if are_nearly_collinear_opposite(p1, p2, p3):
                    return False
    
    return True

"""
Calculate the average step size (distance between consecutive pts) in a segment.
Get pairwise euclidean distances between all adjacent points and take the mean.
"""
def calculate_average_step(segment: list) -> float:
    
    if len(segment) < 2:
        return 1.0
    
    total_distance = sum(distance(segment[i], segment[i+1]) for i in range(len(segment)-1))
    return total_distance / (len(segment) - 1)

"""
Check if three points are nearly collinear but in opposite directions.
Consider three points, p1, p2, p3 and vectors connecting p1 -> p2 and p2 -> p3.
If the vectors point in opposite directions and are separated by at least 143º,
the points are nearly collinear in opposite directions, indicating a sharp reversal
in this path of three points.
"""
def are_nearly_collinear_opposite(p1: tuple, p2: tuple, p3: tuple) -> bool:
    # vector v1: p1 -> p2; vector v2: p2 -> p3
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    
    # Vector magnitudes
    len1 = (v1[0]**2 + v1[1]**2)**0.5
    len2 = (v2[0]**2 + v2[1]**2)**0.5
    
    # Return false for very short vectors to avoid div/0
    if len1 < 1e-10 or len2 < 1e-10:
        return False
    
    # Normalize vectors
    v1_norm = (v1[0]/len1, v1[1]/len1)
    v2_norm = (v2[0]/len2, v2[1]/len2)
    
    # Dot product of normalized vectors
    dot_product = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]
    
    # Check if vectors are nearly opposite (dot product close to -1 [143º])
    return dot_product < -0.8

"""
Calculate Euclidean distance between two points.
"""
def distance(p1: tuple, p2: tuple) -> float:
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
