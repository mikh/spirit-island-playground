# Functions responsible for creating graphics representations

import graphics_api
from Digital_Library.lib import log_lib

PADDING = 100
TIMER_TEST = True
TIMING_ITERATIONS = 20

PERMUTATION_THREADS = 2

THREAD_COUNT_TEST = True
MIN_THREADS = 1
MAX_THREADS = 100
PERMUTATION_THREADS_TO_TEST = [1,2,3,4,5, 7, 10, 15, 25, 50, 100]   #range(MIN_THREADS, MAX_THREADS)

CENTERPOINT_RADIUS_OFFSET = 25

NUM_SUBPOINTS = 700
SUBPOINT_RADIUS = 20

PERMUTATION_TIMER_NAME = 'Best Permutation Selection'
REPORT_NAME = "Spirit-Island-Playground\\Performance Tuning"

def perform_timing_iterations(board, land_distances, width, height, logger, num_permutation_threads=PERMUTATION_THREADS):
    """ Performs timing iterations

        ### Arguments:
            board<board>: board to generate
            land_distances<dict>: mapping of distances between lands
            width<float>: width
            height<float>: height
            logger<QuickLogger>: logger
            num_permutation_threads<int>: number of permutation threads to use
        
        ### Returns:
            all_timers<dict>: timer output
            failure_rate<float>: Percentage failure
            failure_messages<dict>: failure messages
    """
    all_timers = {}
    failure_rate = 0
    failure_messages = {}

    iteration_count = 1
    if TIMER_TEST:
        iteration_count = TIMING_ITERATIONS
    for _ in range(iteration_count):
        timers = voronoi(board.lands, land_distances, (width, height), logger, permutation_threads=num_permutation_threads)
        if not isinstance(timers, dict):
            failure_rate += 1
            if not timers in failure_messages:
                failure_messages[timers] = 0
            failure_messages[timers] += 1
        else:
            for timer in timers:
                if not timer in all_timers:
                    all_timers[timer] = 0
                all_timers[timer] += timers[timer]
    for t in all_timers:
        all_timers[t] /= iteration_count
    return all_timers, failure_rate/iteration_count, failure_messages

def generate_general_report(all_timers, failure_rate, failure_messages):
    """ Generates a general report 

        ### Arguments:
            all_timers<dict>: dicionary of timers
            failure_rate<float>: number of failures
            failure_messages<dict>: failure messages and their counts
    """
    report = log_lib.ReportGenerator(REPORT_NAME)
    report.set_title("General Timing Test on {timestamp}", add_timestamp=True)

    report.add_break("Results")
    for t in all_timers:
        report.add_result(t, "{:.2f}s".format(all_timers[t]))
    report.add_result("Success Rate", "{:.2%}".format(1 - failure_rate))

    if len(failure_messages) > 0:
        report.add_break("Failure Messages")
        for message in failure_messages:
            report.add_result(message, str(failure_messages[message]))
    report.save()


    


def perform_permutation_thread_test(board, land_distances, width, height, logger):
    """ Performs the thread test for permutations

        ### Arguments:
            board<board>: board to generate
            land_distances<dict>: mapping of distances between lands
            width<float>: width
            height<float>: height
            logger<QuickLogger>: logger

    """
    thread_timers = {}
    for thread_count in PERMUTATION_THREADS_TO_TEST:
        timers, failure_rate, failure_messages = perform_timing_iterations(board, land_distances, width, height, logger, num_permutation_threads=thread_count)
        thread_timers[thread_count] = {'timers': timers[PERMUTATION_TIMER_NAME], 'failure_rate': failure_rate, 'failure_messages': failure_messages}

    report = log_lib.ReportGenerator(REPORT_NAME)
    report.set_title("Permutation Timing Test on {timestamp}", add_timestamp=True)

    failure_messages = {}

    report.add_break("Results")
    for t in PERMUTATION_THREADS_TO_TEST:
        result = thread_timers[t]
        s = '{:.2f}s'.format(result['timers'])
        if result['failure_rate'] > 0:
            s += " [Failure Rate: {:.2%}]".format(result['failure_rate'])
        report.add_result(t, s)
        if len(result['failure_messages']) > 0:
            for message in result['failure_messages']:
                if not message in failure_messages:
                    failure_messages[message] = 0
                failure_messages[message] += result['failure_messages'][message]
    
    if len(failure_messages) > 0:
        report.add_break("Failure Messages")
        for message in failure_messages:
            report.add_result(message, str(failure_messages[message]))
    report.save()
    
    

    

def voronoi_generation(width, height, logger, boards):
    """ Creates graphics image in vornoi style

        ### Arguments:
            width<float>: width
            height<float>: height
            logger<QuickLogger>: logger
            boards<list<board>>: boards to generate
    """
    logger.start_section("Running Voronoi Generation with dimensions {}x{}".format(width, height), timer_name="full_gen", end='\n')

    for board_letter in boards:
        logger.start_section("Creating board {}".format(board_letter), end='\n', timer_name="single_board_gen", indent_level=1)

        board = boards[board_letter]
        land_distances = board.get_land_distances()

        #perform_permutation_thread_test(board, land_distances, width, height, logger)
        timers, failure_rate, failure_messages = perform_timing_iterations(board, land_distances, width, height, logger)
        generate_general_report(timers, failure_rate, failure_messages)

        logger.end_section(timer_name='single_board_gen', indent_level=1)
    logger.end_section(timer_name="full_gen")


def voronoi(lands, land_distances, dimensions, logger, permutation_threads=PERMUTATION_THREADS):
    
    """ Performs voronoi calculations

        ### Arguments:
            lands<list<land>>: lands on board
            land_distances<dict>: distances between lands
            dimensions<tuple<float>>: full dimensions
            logger<QuickLogger>: logger

        ### Returns:
            timers<dict>: timers
    """
    timers = {}

    allcoords = []

    boundActive, N, max_radius = init(logger, dimensions, lands)

    centerpoints, timers['Generate Centerpoints']  = graphics_api.generate_points(boundActive, N, max_radius-CENTERPOINT_RADIUS_OFFSET, logger)
    if centerpoints is None:
        logger.info("Fatal error generating centerpoints")
        return "Failed to generate centerpoints"
    
    allcoords.extend(centerpoints)
    
    timers['Initial Centerpoint Selection'] = graphics_api.select_top_left_point(centerpoints, logger)
    centerpoints, timers['Best Permutation Selection'] = graphics_api.calculate_order(centerpoints, N, land_distances, logger, thread_count=permutation_threads)
    
    boundSubpoints = graphics_api.make_bounding_rectangle(dimensions, int(PADDING/2))
    subpoints, timers['Generate SubPoints'] = graphics_api.generate_points(boundSubpoints, NUM_SUBPOINTS, SUBPOINT_RADIUS, logger, existing_points=allcoords)
    if subpoints is None:
        return "Failed to generate subpoints"

    return timers


def init(logger, dimensions, lands, indent_level=2):
    """ Performs init functions

        ### Arguments:
            logger<QuickLogger>: logger
            dimensions<tuple<float>>: full dimensions
            lands<list<land>>: lands on board
            indent_level<int>: logger indent level

        ### Returns:
            rect<BoundingRectangle>: bounding rectangle
            N<int>: number of centerpoints
            max_radius<float>: closest centerpoints can be to each other
    """
    logger.start_section("Initializing Voronoi calculations", indent_level=indent_level, timer_name="init")
    rect = graphics_api.make_bounding_rectangle(dimensions, PADDING)
    N = len(lands)
    max_radius = graphics_api.calculate_max_radius(rect, N)
    logger.end_section(timer_name="init")
    return rect, N, max_radius