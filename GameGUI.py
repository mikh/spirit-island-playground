# Contains the display code

import arcade
import math
from timeit import default_timer as timer
import numpy as np

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
SCREEN_TITLE = 'Spirit Island Playground'

PADDING = 100
SPACING = 100
SQUARE_SIZE = 100
STEP_SIZE = 1

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

    def setup(self, G, padding=PADDING, spacing=SPACING, sq_size=SQUARE_SIZE):
        """ Sets up gui

            ### Arguments:
                self<GameGUI>: self-reference
                G<game>: game reference
        """
        self.shape_list = arcade.ShapeElementList()
        self.connection_list = arcade.ShapeElementList()

        self.land_coords = {}
        self.connections = []
        self.last_land_coords = [None, None]
        self.restrictions = []

        for board_letter in G.boards:
            board = G.boards[board_letter]
            land_distances = board.get_land_distances()
            
            for land_number in range(len(board.lands)):
                land = board.lands[land_number]
                land_coords = self.get_land_coords(padding, spacing, sq_size, land, land_distances)
                
                self.shape_list.append(self.generate_land(land, sq_size, land_coords))

        for connection in self.connections:
            X, Y = connection
            line = arcade.create_line(self.land_coords[X][0], self.land_coords[X][1], self.land_coords[Y][0], self.land_coords[Y][1], arcade.color.RED, 3)
            self.connection_list.append(line)

    def on_draw(self):
        """ Draws the GUI

            ### Arguments:
                self<GameGUI>: self-reference
        """
        arcade.start_render()
        self.connection_list.draw()
        self.shape_list.draw()

        for land in self.land_coords:
            arcade.draw_text(str(land), self.land_coords[land][0] + 5, self.land_coords[land][1] + 5, arcade.color.BLACK, 14)

def launch(G):
    """ Launches the GUI 
        
        G<game>: game reference to build
    """
    gameGUI = GameGUI(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    gameGUI.setup(G)
    arcade.run()
        