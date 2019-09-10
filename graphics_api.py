# library of graphics helper functions

import math
from timeit import default_timer as timer
import random
import threading
import numpy as np

PROGRESS_BAR_SIZE = 150
MAX_THREAD_COUNT = 10

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
            for eC in check_points:
                if calc_distance(eC, C) < max_radius:
                    placed = False
                    break
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
    distances = np.zeros((N, N))



def calculate_order(points, N, adj_D, logger, indent_level=2):
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


