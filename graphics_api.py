# library of graphics helper functions

import math
from timeit import default_timer as timer
import random
import threading
import numpy as np
from itertools import permutations

PROGRESS_BAR_SIZE = 175
#MAX_THREAD_COUNT = 100

class BoundingRectangle():
    """ Rectangle of dimensions """

    def __init__(self, x_min, x_max, y_min, y_max):
        """ Constructor

            ### Arguments:
                self<BoundingRectangle>: self-reference
                x_min<float>: minimum X
                x_max<float>: maximum X
                y_min<float>: minimum Y
                y_max<float>: maximum Y
        """
        self.xmin = x_min
        self.xmax = x_max
        self.ymin = y_min
        self.ymax = y_max

def make_bounding_rectangle(full_dimensions, padding):
    """ Creates a bounding rectangle 

        ### Arguments:
            full_dimensions<tuple>: base dimensions
            padding<float>: inner padding

        ### Returns:
            rectangle<BoundingRectangle>: rectangle of bounds
    """
    return BoundingRectangle(padding, full_dimensions[0] - padding, padding, full_dimensions[1] - padding)

def calculate_max_radius(bound, N):
    """ Calculates the max radius that can be used to fit N points

        ### Arguments:
            bound<BoundingRectangle>: boundaries
            N<int>: number of points

        ### Returns:
            max_radius<float>: max radius
    """
    return math.sqrt((bound.xmax-bound.xmin) * (bound.ymax - bound.ymin)/N)

def calc_distance(coords1, coords2):
    """ Calculates distance between two sets of coords

        ### Arguments:
            coords1<tuple<float>>: first set of coords, X, Y
            coords2<tuple<float>>: second set of coords X, Y
        
        ### Returns:
            distance<float>: distance between coords
    """
    return math.sqrt(math.pow(coords1[0] - coords2[0], 2) + math.pow(coords1[1] - coords2[1], 2))

def generate_points(bound, N, max_radius, logger, indent_level=2, existing_points=None, timeout=30):
    """ Generates points randomly

        ### Arguments:
            bound<BoundingRectangle>: boundaries
            N<int>: number of points to generate
            max_radius<float>: closest two points can be to each other
            logger<QuickLogger>: logger
            indent_level<int>: logger indentation
            existing_points<list<list<float>>>: existing points to account for
            timeout<int>: timeout in seconds 

        ### Returns:
            points<list<list<float>>>: generated points. Does not contain existing points
            elapsed_time<float>: elapsed time
    """
    check_points = []
    if existing_points is not None:
        check_points = existing_points.copy()
    points = []
    start_time = timer()
    for ii in range(N):
        logger.progress(ii/N, "Creating point #{} [{:.2f}s elapsed]...".format(ii, timer()-start_time), indent_level=indent_level, total_size=PROGRESS_BAR_SIZE)

        placed = False
        while not placed:
            if timer() - start_time > timeout: # timeout
                return None, timer() - start_time
            placed = True

            C = [random.randint(bound.xmin, bound.xmax), random.randint(bound.ymin, bound.ymax)]
            if len(check_points) > 0:
                CP = np.array(check_points)
                D = np.sqrt( np.power(CP[:, 0] - C[0], 2) + np.power(CP[:, 1] - C[1], 2))
                sD = sum(D[D<max_radius])
                if sD > 0:
                    placed = False
            if placed:
                points.append(C)
                check_points.append(C)       

    logger.progress(1, "Done. Created {} points [{:.2f}s elapsed]".format(N, timer()-start_time), indent_level=indent_level, total_size=PROGRESS_BAR_SIZE, finish=True)
    return points, timer()-start_time

def select_top_left_point(points, logger, indent_level=2):
    """ Selects the top left point and places it at the start of the points list

        ### Arguments:
            point<list<float>>: list of point coords
            logger<QuickLogger>: logger
            indent_level<int>: indent level for logger

        ### Returns:
            elapsed_time<float>: elapsed time
    """
    logger.start_section("Selecting top-left point", indent_level=indent_level, timer_name="top_left_select")
    best_D = None
    best_C = None
    for C in points:
        D = calc_distance(C, (0,0))
        if best_D is None or best_D > D:
            best_D = D
            best_C = C
    del points[points.index(best_C)]
    points.insert(0, best_C)
    logger.end_section(timer_name="top_left_select")
    return logger.get_delta("top_left_select")

def calculate_all_distances(points, N):
    """ Calculates distances between the first N points in points

        ### Arguments:
            points<list<list<float>>>: points
            N<int>: number of points

        ### Returns:
            distances<list<list<float>>>: list of distances
    """
    distances = np.zeros((N, N))
    for ii in range(N-1):
        C = points[ii]
        TP = np.array(points[ii+1:N])
        D = np.sqrt(np.power(TP[:, 0] - C[0], 2) + np.power(TP[:,1] - C[1], 2))
        distances[ii, ii+1:N] = D
        distances[ii+1:N, ii] = D

        #for jj in range(ii, N):
        #    if ii != jj:
        #        distances[ii][jj] = calc_distance(points[ii], points[jj])
    return distances

def calculate_order(points, N, adj_D, logger, indent_level=2, thread_count=10):
    """ Calculates ideal order of points based on adjacency

        ### Arguments:  
            points<list<float>>: list of point coords
            N<int>: number of points
            adj_D<dict>: adjacency list
            logger<QuickLogger>: logger
            indent_level<int>: indent level for logger

        ### Returns:
            points<dict>: dictionary of points
            elapsed_time<float>: elapsed time
    """
    def print_permutation(perm):
        """ Returns string representation of a permutation

            ### Arguments:
                perm<list<int>>: permutation
            
            ### Returns:
                perm<string>: permutation
        """
        return "[{}]".format(", ".join([str(x) for x in perm]))
    def calculate_permutation_distance(perm, distances, land_distances):
        """ Calculates the permutation distance

            ### Arguments:
                perm<list<int>>: permutation
                distances<list<list<float>>>: distance matrix
                land_distances<dict<dict<int>>>: adjacency list
            
            ### Returns:
                D<float>: total distance
        """
        D = 0
        for ii in range(len(perm)):
            P = perm[ii]
            for eP in land_distances[P]:
                if land_distances[P][eP] == 1:
                    D += distances[ii][perm.index(eP)]
        return D    

    def calculate_order_thread_worker(perms, distances, adj_D, permutation_shortlist):
        best_distance = np.inf
        best_permutation = []

        for perm in perms:
            perm = (0, *perm)
            distance = calculate_permutation_distance(perm, distances, adj_D)
            if distance < best_distance:
                best_distance = distance
                best_permutation = perm
        permutation_shortlist.append([best_distance, best_permutation])

    logger.start_section("Calculating optimal order", indent_level=indent_level, timer_name="order")

    logger.start_section("Init calculations", indent_level=indent_level+1, timer_name="order_init")
    distances = calculate_all_distances(points, N)

    all_perms = list(permutations(range(1, N)))
    #instance = 0
    total = len(all_perms)
    #best_distance = np.inf
    #best_permutation = []
    permutation_shortlist = []
    threads = []
    logger.end_section(message=" {} permutations to check. ".format(total), timer_name="order_init")

    step_size = int(len(all_perms)/thread_count)
    for start_index in range(0, len(all_perms), step_size):
        max_size = min([start_index+step_size, len(all_perms)])
        logger.progress(start_index/len(all_perms), "Creating thread for range {}-{}...".format(start_index, max_size), total_size=PROGRESS_BAR_SIZE, indent_level=indent_level+1)
        subperms = all_perms[start_index:max_size]
        threads.append(threading.Thread(target=calculate_order_thread_worker, args=(subperms, distances, adj_D, permutation_shortlist)))
    logger.progress(1, "{} threads created.".format(len(threads)), finish=True, indent_level=indent_level+1, total_size=PROGRESS_BAR_SIZE)

    for ii in range(len(threads)):
        logger.progress(ii/len(threads), "Starting thread {}...".format(ii), total_size=PROGRESS_BAR_SIZE, indent_level=indent_level+1)
        threads[ii].start()
    logger.progress(1, "{} threads started.".format(len(threads)), total_size=PROGRESS_BAR_SIZE, indent_level=indent_level+1, finish=True)

    threads_complete = 0
    while threads_complete < len(threads):
        threads_complete = 0
        for thread in threads:
            if not thread.isAlive():
                threads_complete += 1
        logger.progress(threads_complete/len(threads), "{} threads finished.".format(threads_complete), total_size=PROGRESS_BAR_SIZE, indent_level=indent_level+1)
    logger.progress(1, "All threads finished. {} results.".format(len(permutation_shortlist)), total_size=PROGRESS_BAR_SIZE, indent_level=indent_level+1, finish=True)

    permutation_shortlist.sort(key=lambda x:x[0])
    best_permutation = permutation_shortlist[0][1]


    """
    for perm in all_perms:
        perm = (0, *perm)
        if instance % 100 == 0:
            logger.progress(instance/total, "Analyzing #{}. Best distance={}. Best Permutation={}".format(instance, best_distance, print_permutation(best_permutation)), total_size=PROGRESS_BAR_SIZE, indent_level=indent_level+1)
        instance += 1
        distance = calculate_permutation_distance(perm, distances, adj_D)
        if distance < best_distance:
            best_distance = distance
            best_permutation = perm
    logger.progress(1, "Finished. Best distance={}. Best Permutation={}".format(best_distance, print_permutation(best_permutation)), total_size=PROGRESS_BAR_SIZE, finish=True, indent_level=indent_level+1)

    """

    new_points = [
        {'allcoords_index': x, 'assignment': best_permutation[x], 'coords': points[x]} for x in range(N)
    ]

    logger.end_section(indent_level=indent_level, timer_name="order")
    return new_points, logger.get_delta("order")

def check_intersection(S1, S2):
    """ Checks if two line segments intersect
        https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
        https://martin-thoma.com/how-to-check-if-two-line-segments-intersect/

        ### Arguments:
            S1<list<list<int>>>: first segment
            S2<list<list<int>>>: second segment

        ### Returns:
            intersect<boolean>: True if they intersect
    """
    X1 = S1[0][0]
    Y1 = S1[0][1]
    X2 = S1[1][0]
    Y2 = S1[1][1]
    X3 = S2[0][0]
    Y3 = S2[0][1]
    X4 = S2[1][0]
    Y4 = S2[1][1]

    minP1 = (min([X1, X2]), min([Y1, Y2]))
    maxP1 = (max([X1, X2]), max([Y1, Y2]))
    minP2 = (min([X3, X4]), min([Y3, Y4]))
    maxP2 = (max([X3, X4]), max([Y3, Y4]))

    if (S1[0] == S2[0] and S1[1] == S2[1]) or (S1[0] == S2[1] and S1[1] == S2[0]):
        return True
    elif S1[0] in S2 or S1[1] in S2:
        return False

    if maxP1[0] < minP2[0] or maxP2[0] < minP1[0] or maxP1[1] < minP2[1] or maxP2[1] < minP1[1]:
        return False
    
    DX1 = X1-X2
    DX2 = X3-X4

    if DX1 == 0 or DX2 == 0:
        if DX1 == DX2:
            return True
        if DX1 == 0:
            Xmatch = X1
            A = (Y3-Y4)/DX2
            b = Y3 - (A*X3)
            Yrange = [minP1[1], maxP1[1]]
        elif DX2 == 0:
            Xmatch = X3    
            A = (Y1-Y2)/DX1
            b = Y1 - (A*X1)
            Yrange = [minP2[1], maxP2[1]]
        Ymatch = A*Xmatch + b
        if Ymatch > Yrange[0] and Ymatch < Yrange[1]:
            return True
        return False
    else:
        A1 = (Y1-Y2)/DX1
        A2 = (Y3-Y4)/DX2
        b1 = Y1 - (A1*X1)
        b2 = Y3 - (A2*X3)
        if A1 == A2:
            if b1 == b2:
                return True
            return False

        Xa = (b2-b1) / (A1-A2)
        if Xa < max([min([X1, X2]), min([X3, X4])]) or Xa > min([max([X1, X2]), max([X3, X4])]):
            return False
        else:
            return True


def matches_any_connection(C, connection_list, coords):
    """ Checks if there is a match against any connection"""
    f_new_l = [coords[C[0]], coords[C[1]]]
    for e_l in connection_list:
        if check_intersection(f_new_l, [coords[e_l[0]], coords[e_l[1]]]):
            return True
    return False

def generate_connection_list(coords, logger, num_connections=8, source_coords=None, existing_connections=None, check_intersections=True, source_coord_exclusivity=False, indent_level=2):
    """ Generates a connection list between source_coords

        ### Arguments:
            coords<list<list<float>>>: list of coords
            logger<QuickLogger>: logger
            num_connections<int>: number of connections to use
            source_coords<list<int>>: list of indices of coords to connect. If None, connects all coords
            existing_connections<list<list<int>>>: list of connections to check against
            check_intersections<boolean>: if True, makes sure no connections intersect
            source_coord_exclusivity<boolean>: if True, source coords can't connect to each other
        
        ### Returns:
            connection_list<list>: list of connections
            elapsed_time<float>: elapsed time
    """
    intersected_connections = 0
    connection_list = []
    if existing_connections is not None:  
        connection_list = existing_connections.copy()
    distances = calculate_all_distances(coords, len(coords))
    source = range(len(coords))
    if source_coords is not None:
        source = source_coords

    start_time = timer()
    instance = 0
    total = len(source)
    for P in source:
        logger.progress(instance/total, "Generating edges for {}. {} cc. {} ic. [{:.2f}s elapsed]".format(P, len(connection_list), intersected_connections, timer() - start_time), indent_level=indent_level, total_size=PROGRESS_BAR_SIZE)
        D = distances[P, :]
        E = sorted([[D[ii], ii] for ii in range(len(D))], key=lambda x:x[0])
        connections_obtained = 0
        ii = 1
        while connections_obtained < num_connections:
            if not source_coord_exclusivity or not E[ii][1] in source:
                C = sorted([P, E[ii][1]])
                if not C in connection_list:
                    if not check_intersections or not matches_any_connection(C, connection_list, coords):
                        connection_list.append(C)
                    else:
                        intersected_connections += 1
                connections_obtained += 1
            ii += 1
    logger.progress(1, "Edges generated. {} cc total. {} ic total. [{:.2f}s elapsed]".format(len(connection_list), intersected_connections, timer() - start_time), indent_level=indent_level, total_size=PROGRESS_BAR_SIZE, finish=True)
    return connection_list, timer() - start_time

