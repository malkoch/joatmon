from gym.envs.registration import register

from joatmon.game.sokoban import *

register(
    'Sokoban-v0',
    entry_point='joatmon.game.sokoban:SokobanEnv',
    kwargs={
        'xml': 'default.xml',
        'xmls': 'game/assets/sokoban/xmls',
        'sprites': 'game/assets/sokoban/sprites'
    }
)
