#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 11:33:11 2025

@author: alexneupauer
"""

from collections import defaultdict

def connect_segments(segments, threshold=5.0, max_step_ratio=3.0):
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


def build_connection_graph(segments, threshold, max_step_ratio):
    """Build a graph of valid connections between segments"""
    
    connections = defaultdict(list)
    
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            # Check all possible ways these segments could connect
            valid_connections = find_valid_connections(
                segments[i], segments[j], i, j, threshold, max_step_ratio
            )
            
            for conn in valid_connections:
                connections[i].append(conn)
                # Add reverse connection
                reverse_conn = reverse_connection(conn)
                connections[j].append(reverse_conn)
    
    return connections


def find_valid_connections(seg1, seg2, idx1, idx2, threshold, max_step_ratio):
    """Find all valid ways two segments can connect"""
    
    valid_connections = []
    
    # Get endpoints
    start1, end1 = seg1[0], seg1[-1]
    start2, end2 = seg2[0], seg2[-1]
    
    # Calculate average step sizes for validation
    avg_step1 = calculate_average_step(seg1)
    avg_step2 = calculate_average_step(seg2)
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
        
        if dist <= threshold and dist <= max_allowed_step:
            valid_connections.append({
                'target': target_idx,
                'type': conn_type,
                'distance': dist,
                'needs_reverse': needs_reverse
            })
    
    return valid_connections


def reverse_connection(conn):
    """Create the reverse connection for bidirectional graph"""
    
    type_map = {
        'end_to_start': 'start_to_end',
        'end_to_end': 'end_to_end',
        'start_to_start': 'start_to_start',
        'start_to_end': 'end_to_start'
    }
    
    return {
        'target': conn['target'] if 'target' in conn else None,
        'type': type_map.get(conn['type'], conn['type']),
        'distance': conn['distance'],
        'needs_reverse': conn['needs_reverse']
    }


def build_optimal_chain(segments, connections, start_seg, used):
    """Build the optimal chain starting from start_seg"""
    
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
    
    # Mark segments as used
    used.update(best_used)
    
    return best_chain if best_chain else list(segments[start_seg])


def extend_chain(segments, connections, start_seg, global_used, direction):
    """Extend chain in one direction"""
    
    chain = list(segments[start_seg])
    used = {start_seg}
    current_seg = start_seg
    
    while True:
        best_connection = None
        best_distance = float('inf')
        
        # Find the best valid connection
        for conn in connections[current_seg]:
            target_seg = conn['target']
            
            if target_seg in used or target_seg in global_used:
                continue
            
            # Check if this connection type is valid for our direction
            if not is_valid_direction(conn['type'], direction):
                continue
            
            if conn['distance'] < best_distance:
                best_connection = conn
                best_distance = conn['distance']
        
        if not best_connection:
            break
        
        # Apply the connection
        target_seg = best_connection['target']
        new_chain = apply_connection(chain, segments[target_seg], best_connection, direction)
        
        if new_chain and is_valid_chain_extension(chain, new_chain):
            chain = new_chain
            used.add(target_seg)
            current_seg = target_seg
        else:
            break
    
    return chain, used


def extend_chain_bidirectional(segments, connections, start_seg, global_used):
    """Extend chain in both directions"""
    
    chain = list(segments[start_seg])
    used = {start_seg}
    
    # Extend forward
    forward_chain, forward_used = extend_chain(segments, connections, start_seg, global_used, 'forward')
    
    # Extend backward from the original segment
    backward_extensions = []
    for conn in connections[start_seg]:
        target_seg = conn['target']
        
        if target_seg in forward_used or target_seg in global_used:
            continue
        
        if is_valid_direction(conn['type'], 'backward'):
            temp_chain = apply_connection(list(segments[start_seg]), segments[target_seg], conn, 'backward')
            if temp_chain:
                backward_extensions.append((target_seg, temp_chain, conn))
    
    # Pick the best backward extension and continue from there
    if backward_extensions:
        best_target, best_backward, best_conn = min(backward_extensions, key=lambda x: x[2]['distance'])
        
        # Now extend this backward chain forward
        temp_used = global_used | forward_used | {best_target}
        final_backward, backward_used = extend_chain(segments, connections, best_target, temp_used, 'forward')
        
        # Combine: final_backward + forward_chain[1:] (skip duplicate middle segment)
        if len(forward_chain) > len(segments[start_seg]):
            combined_chain = final_backward + forward_chain[len(segments[start_seg]):]
        else:
            combined_chain = final_backward
        
        return combined_chain, forward_used | backward_used | {best_target}
    
    return forward_chain, forward_used


def is_valid_direction(conn_type, direction):
    """Check if connection type is valid for the given direction"""
    
    if direction == 'forward':
        return conn_type in ['end_to_start', 'end_to_end']
    elif direction == 'backward':
        return conn_type in ['start_to_start', 'start_to_end']
    
    return True


def apply_connection(current_chain, next_segment, connection, direction):
    """Apply a connection between current chain and next segment"""
    
    try:
        conn_type = connection['type']
        needs_reverse = connection.get('needs_reverse', False)
        
        # Prepare the segment to add
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
    
    except (IndexError, TypeError):
        return None
    
    return None


def is_valid_chain_extension(old_chain, new_chain):
    """Validate that the chain extension doesn't create problems"""
    
    if len(new_chain) <= len(old_chain):
        return False
    
    # Check for duplicate consecutive points
    for i in range(len(new_chain) - 1):
        if new_chain[i] == new_chain[i + 1]:
            return False
    
    # Check for extreme direction changes (potential errors)
    if len(new_chain) >= 3:
        # Look at the connection point area
        connection_area = new_chain[len(old_chain)-2:len(old_chain)+2]
        if len(connection_area) >= 3:
            # Check for sharp reversals that might indicate errors
            for i in range(len(connection_area) - 2):
                p1, p2, p3 = connection_area[i], connection_area[i+1], connection_area[i+2]
                if are_nearly_collinear_opposite(p1, p2, p3):
                    return False
    
    return True


def calculate_average_step(segment):
    """Calculate the average step size in a segment"""
    
    if len(segment) < 2:
        return 1.0
    
    total_distance = sum(distance(segment[i], segment[i+1]) for i in range(len(segment)-1))
    return total_distance / (len(segment) - 1)


def are_nearly_collinear_opposite(p1, p2, p3):
    """Check if three points are nearly collinear but in opposite directions"""
    
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    
    # Normalize vectors
    len1 = (v1[0]**2 + v1[1]**2)**0.5
    len2 = (v2[0]**2 + v2[1]**2)**0.5
    
    if len1 < 1e-10 or len2 < 1e-10:
        return False
    
    v1_norm = (v1[0]/len1, v1[1]/len1)
    v2_norm = (v2[0]/len2, v2[1]/len2)
    
    # Dot product of normalized vectors
    dot_product = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]
    
    # Check if vectors are nearly opposite (dot product close to -1)
    return dot_product < -0.8


def distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
