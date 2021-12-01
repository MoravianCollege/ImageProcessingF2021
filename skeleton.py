"""Functions for working with skeletong."""

import numpy as np
import scipy.ndimage as ndi
from .utils import nonzero, cos


__all__ = ["get_straight_skeleton", "collapse_short_edges", "plot_skeleton_lines"]


def get_straight_skeleton(skeleton):
    """
    This performs a BFS search to find the straight skeleton from a skeleton. It first finds all
    end points and branch points and follows all of the pixels from each of those points outward
    till reaching another one.
    """
    lines = []
    to_visit = skeleton.copy()

    # Get all of the special points in the skeleton
    count_neighbors = ndi.convolve(to_visit.view(np.int8), np.ones((3, 3), np.int8), mode='constant') - 1
    all_pts = set(map(tuple, np.transpose(np.where(to_visit & ((count_neighbors == 1) | (count_neighbors > 2))))))  # set of locations

    # For each of the special points search outward till be reach a destination.
    while all_pts:
        y, x = start = all_pts.pop()
        to_visit[y, x] = False  # clear this point, it has now been visited
        destinations = []
        while True:
            neighborhood = to_visit[y-1:y+2, x-1:x+2]
            if not neighborhood.any(): break  # was an isolated pixel

            # Select any neighbor of the point that has not yet been visiteds
            next_y, next_x = np.unravel_index(np.argmax(neighborhood), neighborhood.shape)
            next_y += max(y - 1, 0)
            next_x += max(x - 1, 0)

            to_visit[next_y, next_x] = False  # we visited it
            if (next_y, next_x) in all_pts:
                # Reached a destination
                destinations.append((next_y, next_x))
                y, x = start  # start over to make sure we find everything from that start point
            else:
                # Moving towards a destination
                y, x = next_y, next_x

        # Each destination represents a line from the start
        # We also need to "unvisit" the destinations back so they can be found again
        for destination in destinations:
            lines.append((start, destination))
            to_visit[destination] = True
    
    # Return all of the line segments
    return np.array(lines)


def collapse_short_edges(lines, min_length, min_length_for_endpoint=0):
    """
    Collapse short edges/lines by merging the endpoints of the line into a single point. This will
    remove endpoints if they are less than the second minimum length. Typically this will be much,
    much, shorter than the min_length and defaults to 0 (never removes them).

    NOTE: A different choice could be one that collapses edges part of triangles with tiny areas.
    This not only incorporates distance but also angle. Close-by points will collapse and points
    that cause very small bends will be collapsed.
    """
    # It will be easier to have a formal graph structure
    graph = {}  # keys are nodes, values are sets of nodes they are connected to
    node_to_point = {}  # node names to actual point value (plus the "weight" of the node so we can average properly)
    point_to_node = {}  # point values to node names (only needed for creating the graph)
    for pt0, pt1 in lines:
        pt0, pt1 = tuple(pt0), tuple(pt1)
        for pt in (pt0, pt1):
            if pt not in point_to_node:
                name = len(point_to_node)
                point_to_node[pt] = name
                node_to_point[name] = np.array(pt + (1,))
        graph.setdefault(point_to_node[pt0], set()).add(point_to_node[pt1])
        graph.setdefault(point_to_node[pt1], set()).add(point_to_node[pt0])

    # Process the graph
    last_num = -1
    while len(graph) != last_num:  # repeat until no changes have been made
        last_num = len(graph)
        for a, adjacent in graph.copy().items():
            if a not in node_to_point: continue  # may have been removed
            pt_a = node_to_point[a]
            for b in adjacent.copy():
                pt_b = node_to_point[b]
                # Check for distance and angle
                this_min_length = min_length_for_endpoint if len(adjacent) == 1 or len(graph[b]) == 1 else min_length
                if (np.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1]) < this_min_length):
                    # Merge node and node2 in the graph
                    for c in graph[b]:
                        graph[c].remove(b)
                        if a != c: graph[c].add(a)
                    graph[a] |= graph[b]
                    graph[a].remove(a)
                    del graph[b]

                    # Update the point location
                    weight = pt_a[2]+pt_b[2]
                    node_to_point[a] = pt_a = np.array((  # the weighted average of the 2 points
                        (pt_a[0]*pt_a[2] + pt_b[0]*pt_b[2]) / weight,
                        (pt_a[1]*pt_a[2] + pt_b[1]*pt_b[2]) / weight, weight))
                    del node_to_point[b]

    # Convert the graph back into lines
    lines = []
    for node, adjacent in graph.items():
        for node2 in adjacent:
            if node < node2:
                lines.append((node_to_point[node][:2], node_to_point[node2][:2]))
    return np.array(lines)


def plot_skeleton_lines(lines, im=None):
    """
    Plots the lines of a skeleton in red along with blue dots for end poitns and green dots for
    branch points.
    """

    from collections import Counter
    import matplotlib.pyplot as plt

    # Draw lines in red
    for pt0, pt1 in lines:
        plt.plot([pt0[1], pt1[1]], [pt0[0], pt1[0]], 'r')

    # Count up each time a point is present in the lines
    # If it only shows up once it is an end point (blue), otherwise it is a branch point (green)
    c = Counter(tuple(pt) for pt, _ in lines)
    c.update(tuple(pt) for _, pt in lines)
    end_pts = np.transpose([pt for pt, count in c.items() if count == 1])
    branch_pts = np.transpose([pt for pt, count in c.items() if count > 1])
    plt.scatter(end_pts[1], end_pts[0], c='b')
    plt.scatter(branch_pts[1], branch_pts[0], c='g')

    # Show the image if provided
    if im is not None:
        plt.imshow(im)
