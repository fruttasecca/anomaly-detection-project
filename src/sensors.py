from aenum import Enum
"""
aenum is useful for accessing data both from string and alias
"""

class Sensors(Enum):
    """
    Sensors.PORTATA1.value returns 0
    Sensors('SO01_01-aMisura').value returns 0
    Sensors.SENSORS_NUM.value returns number of sensors ()
    Sensors.OXY1.string returns 'SO03_01-aMisura'
    """

    _init_ = 'value string'
    PORTATA1 = 0, 'SO01_01-aMisura'
    PORTATA2 = 1, 'SO01_02-aMisura'
    PORTATA3 = 2, 'SO01_03-aMisura'
    SOLIDI1 = 3, 'SO01_04-aMisura'
    SOLIDI2 = 4, 'SO01_05-aMisura'
    SOLIDI3 = 5, 'SO01_06-aMisura'
    OXY1 = 6, 'SO03_01-aMisura'
    OXY2 = 7, 'SO05_01-aMisura'
    OXY3 = 8, 'SO07_01-aMisura'
    AMMO1 = 9, 'SO03_04-aMisura'
    AMMO2 = 10, 'SO05_04-aMisura'
    AMMO3 = 11, 'SO07_04-aMisura'
    NITRATI1 = 12, 'SO03_05-aMisura'
    NITRATI2 = 13, 'SO05_05-aMisura'
    NITRATI3 = 14, 'SO07_05-aMisura'
    SOFFIANTI1 = 15, 'CR03_01-aAssorbimento'
    SOFFIANTI2 = 16, 'CR03_02-aAssorbimento'
    SOFFIANTI3 = 17, 'CR03_03-aAssorbimento'
    SOFFIANTI4 = 18, 'CR03_04-aAssorbimento'
    SOFFIANTI5 = 19, 'CR03_05-aAssorbimento'
    SOFFIANTI6 = 20, 'CR03_06-aAssorbimento'
    VALVE1 = 21, 'SO03_07-aMisura'
    VALVE2 = 22, 'SO05_07-aMisura'
    VALVE3 = 23, 'SO07_07-aMisura'
    SENSORS_NUM = 24, ''

    def __str__(self):
        return self.value[1]

    @classmethod
    def _missing_value_(cls, value):
        for member in cls:
            if member.string == value:
                return member