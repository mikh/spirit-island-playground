# Contains the display code

import arcade
import math
import random
from timeit import default_timer as timer
import numpy as np
from itertools import permutations

from Digital_Library.lib import math_lib
from Digital_Library.lib import log_lib

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
SCREEN_TITLE = 'Spirit Island Playground'

PADDING = 100
SPACING = 100
SQUARE_SIZE = 100
STEP_SIZE = 1

def _calculate_N_and_radius(lands, x_max, x_min, y_max, y_min):
    """ Calculates number of nodes and max radius

        ### Arguments:
            lands<list<land>>: list of lands
            x_max<float>: max x position
            x_min<float>: min x position
            y_max<float>: max y position
            y_min<float>: min y position

        ### Returns:
            N<int>: number of nodes
            max_radius<float>: max radius value
    """
    N = len(lands)
    max_radius = math.sqrt((x_max - x_min) * (y_max - y_min)/N)
    return N, max_radius

def _generate_points(max_radius, x_min, x_max, y_min, y_max, N, logger, existing_points=None):
    """ Generates points randomly

        ### Arguments:
            max_radius<float>: maximum radius to allow
            x_max<float>: max x position
            x_min<float>: min x position
            y_max<float>: max y position
            y_min<float>: min y position
            N<int>: number of points to generate
            logger<QuickLogger>: logger
            existing_points<list<list<float>>>: existing points to account for

        ### Returns:
            coords<list<list<float>>>: list of point coordinates
    """

    if existing_points is not None:
        coords = existing_points.copy()
    else:
        coords = []

    for ii in range(N):
        logger.progress(ii/N, "Creating point #{}".format(ii+1), indent_level=5, total_size=120)
        
        placed = False
        while not placed:
            placed = True
            C = [random.randint(x_min, x_max), random.randint(y_min, y_max)]
            for eC in coords:
                if _calc_distance(eC, C) < max_radius:
                    placed = False
                    break
            if placed:
                coords.append(C)
    logger.progress(1, "Done. Created {} points.".format(N), indent_level=5, total_size=120, finish=True)
    return coords

def _calc_distance(coords1, coords2):
        """ Calculates distance between two sets of coords

            ### Arguments:
                coords1<tuple<float>>: first set of coords, X, Y
                coords2<tuple<float>>: second set of coords X, Y
            
            ### Returns:
                distance<float>: distance between coords
        """
        return math.sqrt(math.pow(coords1[0] - coords2[0], 2) + math.pow(coords1[1] - coords2[1], 2))

def _get_top_left_point(coords):
    best_D = None
    best_C = None
    for C in coords:
        D = _calc_distance(C, [0,0])
        if best_D is None or best_D > D:
            best_D = D
            best_C = C
    return best_C

def _calculate_distances(coords, N):
    distances = np.zeros((N, N))
    for ii in range(N):
        for jj in range(N):
            if ii != jj:
                distances[ii][jj] = _calc_distance(coords[ii], coords[jj])
    return distances

def _print_permutation(perm):
    return "[{}]".format(", ".join([str(x) for x in perm]))

def _calculate_permutation_distance(perm, distances, land_distances):
    D = 0
    for ii in range(len(perm)):
        P = perm[ii]
        for eP in land_distances[P]:
            if land_distances[P][eP] == 1:
                D += distances[ii][perm.index(eP)]
    return D    

def _calculate_order(coords, N, land_distances, logger):
    logger.start_section("Calculating optimal order", indent_level=5, timer_name="calc_order", end='\n')

    logger.start_section("Init calculations", indent_level=6, timer_name='init_calcs')
    distances = _calculate_distances(coords, N)
    order = list(range(N))

    all_perms = list(permutations(range(1, N)))
    instance = 0 
    total = len(all_perms)

    best_distance = np.inf
    best_permutation = []

    logger.end_section(message=" {} Permtations".format(total), timer_name="init_calcs")

    for perm in all_perms:
        perm = (0, *perm)
        if instance % 100 == 0:
            logger.progress(instance/total, "Analyzing #{}. Best distance={}. Best Permutation={}".format(instance, best_distance, _print_permutation(best_permutation)), total_size=170, indent_level=6)
        instance += 1

        distance = _calculate_permutation_distance(perm, distances, land_distances)
        if distance < best_distance:
            best_distance = distance
            best_permutation = perm
    logger.progress(1, "Finished. Best distance={}. Best Permutation={}".format(best_distance, _print_permutation(best_permutation)), total_size=170, finish=True, indent_level=6)
    
    logger.end_section(indent_level=5, timer_name="calc_order")
    return best_permutation

def _check_intersection(S1, S2):
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

def _matches_any_connection(new_l, connection_list, coords):
    """ Checks if the new line segment matches any connection list line segment

        ### Arguments:
            new_l<list<int>>: line segment indices
            connection_list<list<list<int>>>: list of existing line segments indices
            coords<list<list<float>>>: coordinates

        ### Returns:
            anymatch<boolean>: intersection exists
    """
    f_new_l = [coords[new_l[0]], coords[new_l[1]]]
    for e_l in connection_list:
        if _check_intersection(f_new_l, [coords[e_l[0]], coords[e_l[1]]]):
            return True
    return False

def _generate_connection_list(coords, logger, num_connections=8, source_coords=None, existing_connections=None):
    intersected_connections = 0
    connection_list = []
    if existing_connections is not None:
        connection_list = existing_connections
    distances = _calculate_distances(coords, len(coords))

    source = range(len(coords))
    if source_coords is not None:
        source = source_coords

    for P in source:
        logger.progress(P/len(coords), "Generating edges for {}. {} current connections. {} intersected connections".format(P, len(connection_list), intersected_connections), indent_level=5, total_size=150)
        D = distances[P, :]
        E = sorted([[D[ii], ii] for ii in range(len(D))], key=lambda x: x[0])
        for ii in range(1, num_connections+1):
            C = sorted([P, E[ii][1]])
            if not C in connection_list:
                if not _matches_any_connection(C, connection_list, coords):
                    connection_list.append(C)
                else:
                    intersected_connections += 1
    logger.progress(1, "Edges generated. {} connections total. {} intersected connections".format(len(connection_list), intersected_connections), indent_level=5, total_size=150, finish=True)
    return connection_list 

def _transform_connection_list_to_map(L):
    """ Creates a connection map from a connection list

        ### Arguments:
            L<list<list<int>>>: list of node connections

        ### Returns:
            M<dict<list<int>>>: mapping of connections
    """
    M = {}
    for l in L:
        I1 = l[0]
        I2 = l[1]
        if not I1 in M:
            M[I1] = []
        if not I2 in M[I1]:
            M[I1].append(I2)
        if not I2 in M:
            M[I2] = []
        if not I1 in M[I2]:
            M[I2].append(I1)
    return M
        
def _BFS(si, ei, M, R_L):
    """ Performs a BFS to find ei from ei

        ### Arguments:
            si<int>: start node
            ei<int>: end node
            M<dict>: mapping of connections
            R_L<list<int>>: restricted list

        ### Returns: 
            path<list<int>>: list of nodes to get from si to ei. None if none could be found
    """
    paths = [[si]]
    seen_nodes = [si]

    movement = True
    while movement:
        movement = False

        next_paths = []
        for p in paths:
            cur_node = p[-1]
            for next_node in M[cur_node]:
                if next_node == ei:
                    p.append(ei)
                    return p
                if not next_node in p and not next_node in R_L and not next_node in seen_nodes:
                    next_paths.append([*p, next_node])
                    seen_nodes.append(next_node)
                    movement = True
        paths = next_paths
    return None 
                



    
def _build_paths(M, N, logger):
    """ Builds the paths between pairs of nodes using BFS

        ### Arguments: 
            M<dict<list<int>>>: map of connections
            N<list<dict>>: list of nodes to connect
            logger<QuickLogger>: logger

        ### Returns:
            paths<list<dict>>: list of paths
    """
    attempted_orders = []
    base_order = list(range(len(N)))

    for attempt in range(50):
        order = base_order.copy()
        while order in attempted_orders:
            random.shuffle(order)
        attempted_orders.append(order)
        

        connected_nodes = []
        paths = []

        R_L = [x['index'] for x in N]
        instance = 0
        total = len(N)
        completed_paths = 0
        unmappable_paths = 0

        for ii in order:
            start_node = N[ii]
            logger.progress(instance/total, "Building paths for {}. {} Paths completed. {} Unmappable paths".format(start_node['index'], completed_paths, unmappable_paths), total_size=150, indent_level=5)
            start_I = start_node['index']
            connections = start_node['connections']

            for end_I in connections:
                id = [min([start_I, end_I]), max([start_I, end_I])]
                if id in connected_nodes:   # already connected
                    continue
                P = _BFS(start_I, end_I, M, R_L)
                if P is None:
                    unmappable_paths += 1
                else:
                    R_L.extend(P)
                    R_L = list(set(R_L))
                    connected_nodes.append(id)
                    paths.append(P)
                    completed_paths += 1
        logger.progress(1, "All paths built. {} paths completed. {} unmappable paths".format(completed_paths, unmappable_paths), total_size=150, indent_level=5, finish=True)
        if unmappable_paths == 0:
            break   
    return paths


        


class GameGUI(arcade.Window):
    """ Spirit Island display """

    LAND_COLORS = {
        'OCEAN' : arcade.color.DARK_SKY_BLUE,
        'WETLANDS' : arcade.color.LIGHT_BLUE,
        'JUNGLE' : arcade.color.GREEN,
        'MOUNTAIN' : arcade.color.GRAY,
        'SANDS' : arcade.color.CANARY_YELLOW
    }

    def __init__(self, width, height, title, G):
        """ Constructor

            ### Arguments:
                self<GameGUI>: self-reference
                width<int>: width of screen
                height<int>: height of screen
                title<string>: string title
                G<game>: game to display
        """
        self.G = G
        self.logger = self.G.logger
        self.logger.start_section("Displaying Game", end='\n', timer_name="create_display")
        self.logger.start_section("Constructing GameGUI", timer_name="construct_class", indent_level=1)
        super().__init__(width, height, title)
        self.width = width
        self.height = height
        arcade.set_background_color(arcade.color.DARK_BLUE)

        self.shape_list = None
        self.logger.end_section(timer_name="construct_class")

    def setup(self, padding=PADDING, spacing=SPACING, sq_size=SQUARE_SIZE):
        """ Sets up gui

            ### Arguments:
                self<GameGUI>: self-reference
                padding<float>: padding settings
                spacing<float>: spacing of a value
                sq_size<float>: size of square. Deprecated?
        """
        self.logger.start_section("Setting Up GUI", end='\n', timer_name="gamegui_setup", indent_level=1)

        self.logger.start_section("Initializing containers", timer_name="init_containers", indent_level=2)
        self.shape_list = arcade.ShapeElementList()
        self.all_connections = arcade.ShapeElementList()
        self.linking_connections = arcade.ShapeElementList()

        self.land_coords = {}
        self.connections = []
        self.last_land_coords = [None, None]
        self.restrictions = []
        self.logger.end_section(timer_name="init_containers")

        self.logger.start_section("Creating boards", end='\n', timer_name="create_boards", indent_level=2)
        for board_letter in self.G.boards:
            self.logger.start_section("Creating board {}".format(board_letter), end='\n', timer_name="create_single_board", indent_level=3)
            board = self.G.boards[board_letter]
            land_distances = board.get_land_distances()

            self.voronoi_placements_fix(board.lands, land_distances, padding, spacing)
            self.logger.end_section(timer_name="create_single_board", indent_level=3)
        self.logger.end_section(timer_name="create_boards", indent_level=2)
        self.logger.end_section(timer_name="gamegui_setup", indent_level=1)

    def voronoi_placements_fix(self, lands, land_distances, padding, spacing):
        """ Calculates the board based on Voronoi Algorithm adaptation
            #https://gamedev.stackexchange.com/questions/79049/generating-tile-map

            ### Arguments:
                self<GameGUI>: self-reference
                lands<list<land>>: lands in the board
                land_distances<dict>: distances between lands
                padding<int>: padding used
                spacing<int>: spacing between points
        """
        self.logger.start_section("Performing Vornoi Placements", indent_level=4, end='\n', timer_name='voronoi')

        self.logger.start_section("Setting up initial values", indent_level=5, timer_name="init_values")
        # Calculate initial values
        x_min = padding
        x_max = self.width - padding
        y_min = padding
        y_max = self.height - padding

        N, max_radius = _calculate_N_and_radius(lands, x_max, x_min, y_max, y_min)
        self.logger.end_section(message="N = {}, Max Radius = {}. ".format(N, max_radius), timer_name="init_values")

        # Generating land center points

        coords = _generate_points(max_radius-20, x_min, x_max, y_min, y_max, N, self.logger)

        self.logger.start_section("Finishing Initial Coordinates", indent_level=5, end='\n', timer_name="finish_coords")
        start_C = _get_top_left_point(coords)
        del coords[coords.index(start_C)]
        coords.insert(0, start_C)

        self.logger.info("Coords:", indent_level=6)
        for C in coords:
            self.logger.info("[{},{}]".format(C[0], C[1]), indent_level=7)
            self.shape_list.append(arcade.create_ellipse_filled(C[0], C[1], 10, 10, arcade.color.BLACK))

        self.logger.info("Start Point: [{},{}]".format(start_C[0], start_C[1]), indent_level=6)
        self.shape_list.append(arcade.create_ellipse_filled(start_C[0], start_C[1], 10, 10, arcade.color.RED))
        self.logger.end_section(indent_level=5, timer_name="finish_coords")

        # Calculate best placement
        best_permutation = _calculate_order(coords, N, land_distances, self.logger)
        
        self.order = best_permutation
        self.coords = coords

        # Generate subpoints
        self.subcoords = _generate_points(20, x_min/2, x_max+(x_min/2), y_min/2, y_max+(y_min/2), 700, self.logger, existing_points=coords)
        for ii in range(len(self.subcoords)-1, -1, -1):
            S = self.subcoords[ii]
            if S in coords:
                del self.subcoords[ii]
            else:
                self.shape_list.append(arcade.create_ellipse_filled(S[0], S[1], 1, 1, arcade.color.RED))
        self.allcoords = [*self.coords, *self.subcoords]

        # Ceate connections between subpoints
        self.connection_list = _generate_connection_list(self.allcoords, self.logger, source_coords=range(len(self.coords)), num_connections=8)
        self.connection_list = _generate_connection_list(self.allcoords, self.logger, existing_connections=self.connection_list, source_coords=range(len(self.coords), len(self.allcoords)), num_connections=4)
        self.connection_map = _transform_connection_list_to_map(self.connection_list)
        for connection in self.connection_list:
            P1 = self.allcoords[connection[0]]
            P2 = self.allcoords[connection[1]]
            self.all_connections.append(arcade.create_line(P1[0], P1[1], P2[0], P2[1], arcade.color.GRAY))   

        # Calculate paths between adjacent lands
        node_connections = []
        for o, C in zip(self.order, self.coords):
            index = self.allcoords.index(C)
            if index == -1:
                print("Cannot find index!")
            node_connections.append({'index': index, 'connections': [self.allcoords.index(self.coords[self.order.index(x)]) for x in land_distances[o] if land_distances[o][x] == 1]})
        paths = _build_paths(self.connection_map, node_connections, self.logger)
        for path in paths:
            for ii in range(1, len(path)):
                P1 = self.allcoords[path[ii]]
                P2 = self.allcoords[path[ii-1]]
                self.linking_connections.append(arcade.create_line(P1[0], P1[1], P2[0], P2[1], arcade.color.RED, 3))
        
        # Generate outside points
        self.all_outside_points = []
        outside_points = _generate_points(15, 1, x_min/2, 1, y_max-1, 25, self.logger, existing_points=self.allcoords)
        self.allcoords.extend(outside_points)
        self.all_outside_points.extend(outside_points)
        outside_points = _generate_points(15, x_max-(x_min/2), x_max-1, 1, y_max-1, 25, self.logger, existing_points=self.allcoords)
        self.allcoords.extend(outside_points)
        self.all_outside_points.extend(outside_points)
        outside_points = _generate_points(15, 1, x_max-1, 1, y_min/2, 25, self.logger, existing_points=self.allcoords)
        self.allcoords.extend(outside_points)
        self.all_outside_points.extend(outside_points)
        outside_points = _generate_points(15, 1, x_max-1, y_max-(y_min/2), y_max-1, 25, self.logger, existing_points=self.allcoords)
        self.allcoords.extend(outside_points)
        self.all_outside_points.extend(outside_points)

        for C in self.all_outside_points:
            self.shape_list.append(arcade.create_ellipse_filled(C[0], C[1], 1,1, arcade.color.GREEN))

        self.logger.end_section(indent_level=4, timer_name='voronoi')




    def on_draw(self):
        """ Draws the GUI

            ### Arguments:
                self<GameGUI>: self-reference
        """

        def draw_land_numbers(land_order, point_order, coords, land_number_offset=(15, 15), land_number_color=arcade.color.WHITE, land_number_size=14, point_index_offset=(30,30), point_index_color=arcade.color.ORANGE, point_index_size=14):
            """ Draws the land numbers - both the selected number and the original point number.
                Selected number - number of the land in the game
                point number - index of point in coord list
            
                ### Arguments:
                    land_order<list<int>>: list of land numbers
                    point_order<list<int>>: list of point indices
                    coords<list<list<float>>>: list of point indices
                    land_number_offset<tuple<int>>: offset in px for land number
                    land_number_color<arcade.color>: color of land number value
                    land_number_size<int>: size of land number text
                    point_index_offset<tuple<int>>: offset in px for point index
                    point_index_color<arcade.color>>: color of point index value
                    point_index_size<int>: size of point index text
            """
            for land_number, coord_number, C in zip(land_order, point_order, coords):
                arcade.draw_text(str(land_number), C[0] + land_number_offset[0], C[1] + land_number_offset[1], land_number_color, land_number_size)
                arcade.draw_text(str(coord_number), C[0] + point_index_offset[0], C[1] + point_index_offset[1], point_index_color, point_index_size)       

        arcade.start_render()

        self.shape_list.draw()
        self.all_connections.draw()
        self.linking_connections.draw()

        draw_land_numbers(self.order, range(len(self.order)), self.coords)

    

def launch(G):
    """ Launches the GUI 
        
        G<game>: game reference to build
    """
    gameGUI = GameGUI(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, G)
    gameGUI.setup()

    gameGUI.logger.end_section(timer_name="create_display")
    arcade.run()
        