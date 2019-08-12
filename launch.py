# Entrance script

import os
import json

from Digital_Library.lib import arg_lib

import game

def main(num_boards):
    """ Launcher 

        ### Arguments:
            num_boards<int>: number of boards to play with
    """

    G = game.Game(int(num_boards))
    with open(os.path.join('output', 'game.json'), 'w') as f:
        json.dump(G.asdict(), f, sort_keys=True, indent=4)
    G.display()



if __name__ == "__main__":
    description = 'Creates Spirit Island Simulation'

    set_vars = {
        "num_boards": {"help": "Number of boards to play with", "value": 1}
    }

    flag_vars = {

    }

    arg_controller = arg_lib.ArgumentController(description=description, set_variables=set_vars, flag_variables=flag_vars)
    var_data = arg_controller.parse_args()
    if var_data is not None:
       main(**var_data)

