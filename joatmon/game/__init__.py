from gym.envs.registration import register

from .cube import *
from .puzzle import *
from .sokoban import *
from .tiles import *

register(
    'Sokoban-v0',
    entry_point='joatmon.game.sokoban:SokobanEnv',
    kwargs={
        'xml': 'default.xml',
        'xmls': 'game/assets/sokoban/xmls',
        'sprites': 'game/assets/sokoban/sprites'
    }
)
