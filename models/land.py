# Land Model

from enum import Enum

class LandTypes(Enum):
    """ Land types """

    OCEAN = 0
    WETLANDS = 1
    JUNGLE = 2
    MOUNTAIN = 3
    SANDS = 4

class Land():
    """ Land Class"""

    def __init__(self, number, land_type, num_towns=0, num_cities=0, num_dahan=0, num_blight=0):
        """ Constructor

            ### Arguments:
                self<Land>: self-reference
                number<int>: land number on board
                land_type<LandTypes>: land type
                num_towns<int>: starting number of towns
                num_cities<int>: starting number of cities
                num_dahan<int>: starting number of dahan
                num_blight<int>: starting number of blight
        """
        self.number = number
        self.land_type = land_type
        self.num_towns = num_towns
        self.num_cities = num_cities
        self.num_dahan = num_dahan
        self.num_blight = num_blight
        self.connected_lands = []

    def add_connected_lands(self, connected_lands):
        """ Adds connected lands

            ### Arguments:
                self<Land>: self-reference
                connected_lands<list<Land>>: connected lands to add
        """
        for land in connected_lands:
            if not land in self.connected_lands:
                self.connected_lands.append(land)
                land.add_connected_lands([self])
    
    def asdict(self):
        """ Returns dict version of Land

            ### Arguments:
                self<Land>: self-reference

            ### Returns:
                D<dict>: dictionary version of land
        """
        return {
            'number': self.number,
            'land_type': self.land_type.name,
            'num_towns': self.num_towns,
            'num_cities': self.num_cities,
            'num_dahan': self.num_dahan,
            'num_blight': self.num_blight,
            'connected_lands': [x.number for x in self.connected_lands]
        }

        
        