# Model for board

import models.land as land

class Board():
    """ Board model """

    INITIAL_SETUP = {
        'A': {
            0: {'land_type': land.LandTypes.OCEAN, 'connected_lands': [1, 2, 3]},
            1: {'land_type': land.LandTypes.MOUNTAIN, 'connected_lands': [0, 2, 4, 5, 6]},
            2: {'land_type': land.LandTypes.WETLANDS, 'num_cities': 1, 'num_dahan': 1, 'connected_lands': [0, 1, 3, 4]},
            3: {'land_type': land.LandTypes.JUNGLE, 'num_dahan': 2, 'connected_lands': [0, 2, 4]},
            4: {'land_type': land.LandTypes.SANDS, 'num_blight': 1, 'connected_lands': [1, 2, 3, 5]},
            5: {'land_type': land.LandTypes.WETLANDS, 'connected_lands': [1, 4, 6, 7, 8]},
            6: {'land_type': land.LandTypes.MOUNTAIN, 'num_dahan': 1, 'connected_lands': [1, 5, 8]},
            7: {'land_type': land.LandTypes.SANDS, 'num_dahan': 2, 'connected_lands': [5, 8]},
            8: {'land_type': land.LandTypes.JUNGLE, 'num_towns': 1, 'connected_lands': [5, 6, 7]}
        }
    }

    def __init__(self, board_letter):
        """ Board constructor 

            ### Arguments:
                self<Board>: self-reference
                board_letter<string>: letter for board - currently supported: A
        """
        self.board_letter = board_letter
        self.lands = {}

        setup = self.INITIAL_SETUP[self.board_letter]
        for land_number in setup:
            self.lands[land_number] = land.Land(land_number, **{x:setup[land_number][x] for x in setup[land_number] if x != 'connected_lands'})

        for land_number in setup:
            connected_lands = setup[land_number]['connected_lands']
            self.lands[land_number].add_connected_lands([self.lands[x] for x in connected_lands])

    def asdict(self):
        """ Converts board to dict

            ### Arguments:
                self<Board>: self-reference

            ### Returns:
                D<dict>: board dict
        """
        return {
            'board_letter': self.board_letter,
            'lands': {x:self.lands[x].asdict() for x in self.lands}
        }

    def get_land_distances(self):
        """ Calculates minimum distances from each land to each other land

            ### Arguments:
                self<board>: self-reference
            
            ### Returns: 
                distances<dict>: distances
        """
        size = len(self.INITIAL_SETUP[self.board_letter])
        distances = [[9999 for x in range(size)] for y in range(size)]
        for base in self.INITIAL_SETUP[self.board_letter]:
            distances[base][base] = 0
            con_l = self.INITIAL_SETUP[self.board_letter][base]['connected_lands']
            for conn in con_l:
                distances[base][conn] = 1

        # Floyd-Warshall All-pair shortest paths algorithm
        for k in range(size):
            for i in range(size):
                for j in range(size):
                    if distances[i][j] > distances[i][k] + distances[k][j]:
                        distances[i][j] = distances[i][k] + distances[k][j]

        output = {}
        for k in range(size):
            output[k] = {}
            for i in range(size):
                output[k][i] = distances[k][i]
        return output

        


    
        
            