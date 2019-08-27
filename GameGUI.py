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
    N = len(lands)
    max_radius = math.sqrt((x_max - x_min) * (y_max - y_min)/N)
    return N, max_radius

def _generate_points(max_radius, x_min, x_max, y_min, y_max, N, sub_offset=20, existing_points=None):
    if existing_points is not None:
        coords = existing_points.copy()
    else:
        coords = []

    for ii in range(N):
        placed = False
        while not placed:
            placed = True
            C = [random.randint(x_min, x_max), random.randint(y_min, y_max)]
            for eC in coords:
                if _calc_distance(eC, C) < (max_radius - sub_offset):
                    placed = False
                    break
            if placed:
                coords.append(C)
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

def _calculate_order(coords, N, land_distances):
    distances = _calculate_distances(coords, N)
    order = list(range(N))

    all_perms = list(permutations(range(1, N)))
    instance = 0 
    total = len(all_perms)
    print("Number of permutations: ", total)

    best_distance = np.inf
    best_permutation = []

    for perm in all_perms:
        perm = (0, *perm)
        if instance % 100 == 0:
            log_lib.update_progress_bar(instance/total, "Analyzing #{}. Best distance={}. Best Permutation={}".format(instance, best_distance, _print_permutation(best_permutation)), total_size=170)
        instance += 1

        distance = _calculate_permutation_distance(perm, distances, land_distances)
        if distance < best_distance:
            best_distance = distance
            best_permutation = perm
    log_lib.update_progress_bar(1, "Finished. Best distance={}. Best Permutation={}".format(best_distance, _print_permutation(best_permutation)), total_size=170, end=True)
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

    if(max([X1, X2]) < min([X3, X4]) or (max([Y1, Y2]) < min([Y3, Y4]))):
        return False
    
    DX1 = X1-X2
    DX2 = X3-X4

    if DX1 == 0 or DX2 == 0:
        pass
    else:
        A1 = (Y1-Y2)/DX1
        A2 = (Y3-Y4)/DX2
        b1 = Y1 - (A1*X1)
        b2 = Y3 - (A2*X3)
        if A1 == A2:
            return False
        Xa = (b2-b1) / (A1-A2)
        if Xa < max([min([X1, X2]), min([X3, X4])]) or Xa > 



def _generate_connection_list(coords, num_connections=5):
    connection_list = []
    distances = _calculate_distances(coords, len(coords))

    for P in range(len(coords)):
        D = distances[P, :]
        E = sorted([[D[ii], ii] for ii in range(len(D))], key=lambda x: x[0])
        for ii in range(1, num_connections+1):
            C = sorted([P, E[ii][1]])
            if not C in connection_list:
                connection_list.append(C)

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
                



    
def _build_paths(M, N):
    """ Builds the paths between pairs of nodes using BFS

        ### Arguments: 
            M<dict<list<int>>>: map of connections
            N<list<dict>>: list of nodes to connect

        ### Returns:
            paths<list<dict>>: list of paths
    """
    connected_nodes = []
    paths = []

    R_L = [x['index'] for x in N]

    for start_node in N:
        start_I = start_node['index']
        connections = start_node['connections']

        for end_I in connections:
            id = [min([start_I, end_I]), max([start_I, end_I])]
            if id in connected_nodes:   # already connected
                continue
            P = _BFS(start_I, end_I, M, R_L)
            if P is None:
                print("Could not find path!")
            else:
                print("Path found from {} => {}".format(start_I, end_I))
                R_L.extend(P)
                R_L = list(set(R_L))
                connected_nodes.append(id)
                paths.append(P)
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

    def __init__(self, width, height, title):
        """ Constructor

            ### Arguments:
                self<GameGUI>: self-reference
                width<int>: width of screen
                height<int>: height of screen
                title<string>: string title
        """
        super().__init__(width, height, title)
        self.width = width
        self.height = height
        arcade.set_background_color(arcade.color.DARK_BLUE)

        self.shape_list = None

    def voronoi_placements_fix(self, lands, land_distances, padding, spacing):
        #https://gamedev.stackexchange.com/questions/79049/generating-tile-map

        # Calculate initial values
        x_min = padding
        x_max = self.width - padding
        y_min = padding
        y_max = self.height - padding

        N, max_radius = _calculate_N_and_radius(lands, x_max, x_min, y_max, y_min)
        print("Max Radius: ", max_radius)

        # Generating land center points
        coords = _generate_points(max_radius, x_min, x_max, y_min, y_max, N)
        start_C = _get_top_left_point(coords)
        del coords[coords.index(start_C)]
        coords.insert(0, start_C)
        print("Coords:")
        for C in coords:
            print("\t[{},{}]".format(C[0], C[1]))
            self.shape_list.append(arcade.create_ellipse_filled(C[0], C[1], 10, 10, arcade.color.BLACK))

        print("Start Point: [{},{}]".format(start_C[0], start_C[1]))
        self.shape_list.append(arcade.create_ellipse_filled(start_C[0], start_C[1], 10, 10, arcade.color.RED))

        # Calculate best placement
        best_permutation = _calculate_order(coords, N, land_distances)
        
        self.order = best_permutation
        self.coords = coords

        # Generate subpoints
        self.subcoords = _generate_points(50, x_min, x_max, y_min, y_max, 400, existing_points=coords)
        for ii in range(len(self.subcoords)-1, -1, -1):
            S = self.subcoords[ii]
            if S in coords:
                del self.subcoords[ii]
            else:
                self.shape_list.append(arcade.create_ellipse_filled(S[0], S[1], 1, 1, arcade.color.RED))
        self.allcoords = [*self.coords, *self.subcoords]

        # Ceate connections between subpoints
        self.connection_list = _generate_connection_list(self.allcoords)
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
        paths = _build_paths(self.connection_map, node_connections)
        for path in paths:
            for ii in range(1, len(path)):
                P1 = self.allcoords[path[ii]]
                P2 = self.allcoords[path[ii-1]]
                self.linking_connections.append(arcade.create_line(P1[0], P1[1], P2[0], P2[1], arcade.color.RED, 3))




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

    def setup(self, G, padding=PADDING, spacing=SPACING, sq_size=SQUARE_SIZE):
        """ Sets up gui

            ### Arguments:
                self<GameGUI>: self-reference
                G<game>: game reference
        """
        self.shape_list = arcade.ShapeElementList()
        self.all_connections = arcade.ShapeElementList()
        self.linking_connections = arcade.ShapeElementList()

        self.land_coords = {}
        self.connections = []
        self.last_land_coords = [None, None]
        self.restrictions = []

        for board_letter in G.boards:
            board = G.boards[board_letter]
            land_distances = board.get_land_distances()

            self.voronoi_placements_fix(board.lands, land_distances, padding, spacing)

            


    def setup_old(self):
        #self.voronoi_placements(board.lands, land_distances, padding, spacing)

        #calculate_coords = self.calculate_placements(board.lands, land_distances, padding, spacing, sq_size)
        #for land in calculate_coords:
        #    self.shape_list.append(self.generate_land(board.lands[land], sq_size, [calculate_coords[land][0], calculate_coords[land][1]]))
        
        #for land_number in range(len(board.lands)):
        #    land = board.lands[land_number]
        #    land_coords = self.get_land_coords(padding, spacing, sq_size, land, land_distances)
            
        #    self.shape_list.append(self.generate_land(land, sq_size, land_coords))

        #for connection in self.connections:
        #    X, Y = connection
        #    line = arcade.create_line(self.land_coords[X][0], self.land_coords[X][1], self.land_coords[Y][0], self.land_coords[Y][1], arcade.color.RED, 3)
        #    self.connection_list.append(line)
        pass

    def draw_old(self):
        #land_num = 0
        #for land in self.best_permutation:
        #    arcade.draw_text(str(land_num), self.final_coordinates[land][0] + 15, self.final_coordinates[land][1] + 15, arcade.color.WHITE, 14)
        #    land_num += 1
        #for ii in range(len(self.final_coordinates)):
        #    arcade.draw_text(str(ii), self.final_coordinates[ii][0]+30, self.final_coordinates[ii][1]+30, arcade.color.ORANGE)
        """
        for ii in range(len(self.final_coordinates)):
            for jj in range(ii+1, len(self.final_coordinates)):
                iC = self.final_coordinates[ii]
                jC = self.final_coordinates[jj]
                arcade.draw_line(iC[0], iC[1], jC[0], jC[1], arcade.color.GRAY)
                distance = GameGUI.calc_distance(iC, jC)
                label_coord = [
                    (max([iC[0], jC[0]]) - min([iC[0], jC[0]]))/2 + min([iC[0], jC[0]]),
                    (max([iC[1], jC[1]]) - min([iC[1], jC[1]])/2) + min([iC[1], jC[1]])
                ]
                arcade.draw_text("{:.2f}".format(distance), label_coord[0], label_coord[1], arcade.color.ORANGE)
        """
        pass


    def voronoi_placements(self, lands, land_distances, padding, spacing):
        random.seed(a=10)
        x_min = padding
        x_max = self.width - padding
        y_min = padding
        y_max = self.height - padding

        coords = []

        for ii in range(len(lands)):
            placed = False
            while not placed:
                placed = True
                C = [random.randint(x_min, x_max), random.randint(y_min, y_max)]
                for eC in coords:
                    if GameGUI.calc_distance(eC, C) < (spacing):
                        placed = False
                        break
                if placed:
                    coords.append(C)
                    self.shape_list.append(arcade.create_ellipse_filled(C[0], C[1], 10, 10, arcade.color.BLACK))
        distances = np.zeros((len(coords), len(coords)))
        for ii in range(len(coords)):
            for jj in range(len(coords)):
                if ii == jj:
                    distances[ii][jj] = 0
                else:
                    distances[ii][jj] = GameGUI.calc_distance(coords[ii], coords[jj])


        print("Coords: ")
        for C in coords:
            print("{}\t{}".format(C[0], C[1]))

        L = list(permutations(range(len(lands))))
        print("Number of permutations: {}".format(len(L)))

        best_score = -1
        best_permutation = None

        def calc_score(perm, land_distances, distances):
            #perm = (7,1,2,5,6,8,0,4,3)
            # print("Try this")
            score = 0
            for land in land_distances:
                dist_breakdown = {}
                for nex_land in land_distances[land]:
                    if land != nex_land:
                        D = land_distances[land][nex_land]
                        if not D in dist_breakdown:
                            dist_breakdown[D] = []
                        dist_breakdown[D].append(distances[perm.index(land)][perm.index(nex_land)])
                keys = sorted(list(dist_breakdown.keys()))

                consistent = True
                prev_max = max(dist_breakdown[keys[0]])
                for ii in range(1, len(keys)):
                    cur_min = min(dist_breakdown[keys[ii]])
                    if prev_max > cur_min:
                        consistent = False
                        #break
                    else:
                        score += 1
                    prev_max = max(dist_breakdown[keys[ii]])
                if consistent:
                    score += 5             
            return score
        instance = 0
        total = len(L)
        for perm in L:
            if instance % 1000 == 0:
                log_lib.update_progress_bar(instance/total, "Scanning permutation #{}. Best Score={}...".format(instance, best_score), total_size=200)
            instance += 1
            score = calc_score(perm, land_distances, distances)
            if score > best_score:
                best_score = score
                best_permutation = perm
        log_lib.update_progress_bar(1, "Done. Best Score={}. Best Permutation = [{}]".format(best_score, ", ".join([str(x) for x in best_permutation]) ), total_size=200, end='\n')

        self.best_permutation = best_permutation
        self.final_coordinates = coords

        

        return None

        

    def calculate_placements(self, lands, land_distances, padding, spacing, sq_size):
        x_min = padding
        x_max = self.width - padding
        y_min = padding
        y_max = self.height - padding

        MIN_DISTANCE = spacing
        TRIALS = 10
        STEP_SIZE = 1
        MAX_MOVEMENT = spacing


        coords = {}
        coords[0] = [x_min, int(self.height/2)]
        x_min += (spacing  + sq_size)
        for land in range(1, len(lands)):
            coords[land] = [random.randint(x_min, x_max), random.randint(y_min, y_max)]
        
        prev_values = {}
        print("Initial")
        for ii in range(len(coords)):
            print("\t{}: [{},{}]".format(ii, coords[ii][0], coords[ii][1]))
            prev_values[ii] = coords[ii]
        base = np.array([[[x,y] for y in range(y_min, y_max, STEP_SIZE)] for x in range(x_min, x_max, STEP_SIZE)])
        
        """
        distances = np.zeros((len(coords), len(coords)))
        for ii in range(len(coords)):
            for jj in range(len(coords)):
                if ii == jj:
                    distances[ii][jj] = 0
                else:
                    distances[ii][jj] = GameGUI.calc_distance(coords[ii], coords[jj])
                    if distances[ii][jj] < MIN_DISTANCE:
                        distances[ii][jj] = np.inf
        """
        for trial in range(TRIALS):
            
            land_order = list(range(1, len(coords)))
            random.shuffle(land_order)
            for ii in land_order:
                P = coords[ii]
                P_ymin = max([y_min, P[1] - MAX_MOVEMENT])
                P_xmin = max([x_min, P[0] - MAX_MOVEMENT])
                y_range = range(P_ymin, min([y_max, P[1]+MAX_MOVEMENT]), STEP_SIZE)
                x_range = range(P_xmin, min([x_max, P[0]+MAX_MOVEMENT]), STEP_SIZE)
                move_coords = np.array([[[x,y] for y in y_range] for x in x_range])

                restriction_shortlist = []
                for jj in range(len(coords)):
                    if ii != jj:
                        restriction_shortlist.append({'coords': coords[jj], 'distance': land_distances[ii][jj]*(spacing + sq_size)})
                best_coords = None
                total = np.zeros((move_coords.shape[0], move_coords.shape[1]))
                for restriction in restriction_shortlist:
                    C = restriction['coords']
                    a = np.square(move_coords[:, :, 0] - C[0]) + np.square(move_coords[:, :, 1] - C[1])
                    d = np.sqrt(a)

                    d[d < restriction['distance']] -= 9000
                    d[d < 0] *= (-1)

                    total += d
                best_coords = np.where(total == np.min(total))
                best_coords = (best_coords[0][0] + P_xmin, best_coords[1][0] + P_ymin)

                coords[ii] = best_coords
            print("Trial #{}".format(trial))
            for ii in range(len(coords)):
                print("\t{}: [{},{}] d {:.2f}".format(ii, coords[ii][0], coords[ii][1], GameGUI.calc_distance(coords[ii], prev_values[ii])))
                prev_values[ii] = coords[ii]
            
        return coords

        
         



    def get_land_coords(self, padding, spacing, sq_size, land, land_distances):
        """ Calculates land coords

            ### Arguments:
                self<GameGUI>: self-reference
                padding<int>: padding to use
                spacing<int>: spacing to use
                sq_size<int>: square size
                land<Land>: land coordinates
                land_distances<dict>: distances in hops from one land to another
        """

        def calc_distance(coords1, coords2):
            """ Calculates distance between two sets of coords

                ### Arguments:
                    coords1<tuple<float>>: first set of coords, X, Y
                    coords2<tuple<float>>: second set of coords X, Y
                
                ### Returns:
                    distance<float>: distance between coords
            """
            return math.sqrt(math.pow(coords1[0] - coords2[0], 2) + math.pow(coords1[1] - coords2[1], 2))

        D = land.asdict()
        
        if len(self.restrictions) == 0:
            X = padding
            Y = int(self.height /2)
            self.restrictions.append({'number': D['number'], 'coords': (X,Y)})
        else:
            min_X = self.restrictions[0]['coords'][0] + land_distances[D['number']][self.restrictions[0]['number']]*(spacing + sq_size)
            max_X = self.width - padding
            min_Y = padding
            max_Y = self.height - padding

            restriction_shortlist = []
            for restriction in self.restrictions:
                restriction_shortlist.append({'coords': restriction['coords'], 'distance': land_distances[D['number']][restriction['number']]*(spacing+sq_size)})
            
            best_coords = None

            base = np.array([[[x,y] for y in range(min_Y, max_Y, STEP_SIZE)] for x in range(min_X, max_X, STEP_SIZE)])
            total = np.zeros((base.shape[0], base.shape[1]))
            for restriction in restriction_shortlist:
                C = restriction['coords']
                a = np.square(base[:,:,0] - C[0]) + np.square(base[:,:,1] - C[1])
                d = np.sqrt(a)
                d[d < restriction['distance']] = np.inf
                total += d
            best_coords = np.where(total == np.min(total))
            best_coords = (best_coords[0][0]+min_X, best_coords[1][0]+min_Y)

            if best_coords is not None:
                X = best_coords[0]
                Y = best_coords[1]
                self.restrictions.append({'number': D['number'], 'coords': (X, Y)})
            else:
                print("Could not find location!!!!")
        return X, Y

    def add_connections(self, origin, C):
        """ Adds land connections 

            ### Arguments:
                self<GameGUI>: self-reference
                origin<int>: original land
                C<list<int>>: connectioned lands
        """
        for connection in C:
            if origin != connection:
                NC = [min([origin, connection]), max([origin, connection])]
                if not NC in self.connections:
                    self.connections.append(NC)
        

    def generate_land(self, land, sq_size, coords):
        """ Generates a land rectangle

            ### Arguments:
                self<GameGUI>: self-reference
                land<Land>: land object
                sq_size<int>: square size
                coords<tuple<float>>: tuple of coords

            ### Returns:
                shape<arcade Object>: arcade shape
        """
        D = land.asdict()
        shape = arcade.create_rectangle_filled(coords[0], coords[1], sq_size, sq_size, self.LAND_COLORS[D['land_type']])
        self.land_coords[D['number']] = [coords[0], coords[1]]
        self.add_connections(D['number'], D['connected_lands'])
        return shape

def launch(G):
    """ Launches the GUI 
        
        G<game>: game reference to build
    """
    gameGUI = GameGUI(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    gameGUI.setup(G)
    arcade.run()
        