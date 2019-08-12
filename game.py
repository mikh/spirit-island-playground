# Game Controller

from Digital_Library.lib import log_lib

import models.board as board
import GameGUI

class Game():
    """ Game Controller """

    def __init__(self, num_boards):
        """ Constructor

            ### Arguments:
                self<Game>: self-reference
                num_boards<int>: number of boards to play with
        """
        self.logger = log_lib.QuickLogger("spirit-island-playground")
        self.logger.start_section("Constructing Game", timer_name="constructor", end='\n')

        self.build_boards(num_boards)

        self.logger.end_section(timer_name="constructor")

    def build_boards(self, num_boards):
        """ Builds the game boards 

            ### Arguments:
                self<Game>: self-reference
                num_boards<int>: number of boards to build
        """
        self.logger.start_section("Building Boards", indent_level=1, timer_name="build_boards")

        self.boards = {}
        board_letters = ['A', 'B', 'C', 'D']
        for board_index in range(0, num_boards):
            self.boards[board_letters[board_index]] = board.Board(board_letters[board_index])

        self.logger.end_section(timer_name="build_boards")

    def display(self):
        """ Displays the game as an arcade board 

            ### Arguments:
                self<Game>: self-reference
        """
        GameGUI.launch(self)

    def asdict(self):
        """ Converts Game to dict

            ### Arguments:
                self<Game>: self-reference
            
            ### Returns:
                d<dict>: dictionary version
        """
        return {
            'boards': {x: self.boards[x].asdict() for x in self.boards}
        }
