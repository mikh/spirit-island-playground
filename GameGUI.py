# Contains the display code

import arcade

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
SCREEN_TITLE = 'Spirit Island Playground'

PADDING = 100
SPACING = 100
SQUARE_SIZE = 100

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

    def get_next_land_coords(self, padding, spacing, sq_size):
        """ Calculates next land coords

            ### Arguments:
                self<GameGUI>: self-reference
                padding<int>: padding to use
                spacing<int>: spacing to use
                sq_size<int>: square size
        """
        X, Y = self.last_land_coords
        if X is None and Y is None:
            X = padding
            Y = padding
        else:
            Y += (sq_size + spacing)
            if Y > self.height - padding:
                Y = padding
                X += (sq_size + spacing)

        self.last_land_coords = [X, Y]

    def get_land_coords(self, padding, spacing, sq_size, land):
        """ Calculates land coords

            ### Arguments:
                self<GameGUI>: self-reference
                padding<int>: padding to use
                spacing<int>: spacing to use
                sq_size<int>: square size
                land<Land>: land coordinates
        """
        D = land.asdict()
        restrictions = []

        if D['number'] in self.restrictions:
            restrictions = self.restrictions[D['number']]
        
        X = padding
        Y = padding



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
        

    def generate_land(self, land, sq_size):
        """ Generates a land rectangle

            ### Arguments:
                self<GameGUI>: self-reference
                land<Land>: land object
                sq_size<int>: square size

            ### Returns:
                shape<arcade Object>: arcade shape
        """
        D = land.asdict()
        shape = arcade.create_rectangle_filled(self.last_land_coords[0], self.last_land_coords[1], sq_size, sq_size, self.LAND_COLORS[D['land_type']])
        self.land_coords[D['number']] = [self.last_land_coords[0], self.last_land_coords[1]]
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
        self.restrictions = {}

        for board_letter in G.boards:
            board = G.boards[board_letter]
            land_distances = board.get_land_distances()
            
            for land_number in board.lands:
                land = board.lands[land_number]
                land_coords = self.get_land_coords(padding, spacing, sq_size, land)
                
                self.get_next_land_coords(padding, spacing, sq_size)
                self.shape_list.append(self.generate_land(land, sq_size))

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
        