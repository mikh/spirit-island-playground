# Functions responsible for creating graphics representations

import graphics_api

PADDING = 100
TIMER_TEST = True
TIMING_ITERATIONS = 50

CENTERPOINT_RADIUS_OFFSET = 25

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

        if TIMER_TEST:
            all_timers = {}
            failure_rate = 0
            failure_messages = {}
            for _ in range(TIMING_ITERATIONS):
                timers = voronoi(board.lands, land_distances, (width, height), logger)
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

            print("\n\n")
            print("-"*60)
            print("Timing Test Results:")
            print("")
            if len(failure_messages) > 0:
                print("\tFailures:")
                print("")
                print("\t\tFailure Rate: {:.2%}".format(failure_rate/TIMING_ITERATIONS))
                for failure_message in failure_messages:
                    print("\t\t{} : {}".format(failure_message, failure_messages[failure_message]))

                print("\n\n")
            print("\tTiming Stats:")
            print("")
            for timer in all_timers:
                print("\t\t{} : {:.2f}s".format(timer, all_timers[timer]/TIMING_ITERATIONS))
            print("\n")
            print("-"*60)
            print("\n\n")
        else:
            voronoi(board.lands, land_distances, (width, height), logger)

        logger.end_section(timer_name='single_board_gen', indent_level=1)
    logger.end_section(timer_name="full_gen")


def voronoi(lands, land_distances, dimensions, logger):
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