import marshal
import random
import xml.etree.ElementTree as ElementTree

import cv2
import numpy as np
import pygame
import pymunk
from pygame.color import THECOLORS
from pymunk.body import Body
from pymunk.vec2d import Vec2d

__all__ = ['SokobanEnv']

from joatmon.game.core import CoreEnv

TYPE_LOOKUP = {
    0: 'wall',
    1: 'empty space',
    2: 'box target',
    3: 'box on target',
    4: 'box not on target',
    5: 'player'
}

ACTION_LOOKUP = {
    0: 'push up',
    1: 'push down',
    2: 'push left',
    3: 'push right',
    4: 'move up',
    5: 'move down',
    6: 'move left',
    7: 'move right',
}

CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}


def generate_room(dim=(7, 7), wall_prob=0.3, p_change_directions=0.35, num_steps=5, num_boxes=1, tries=4):
    def room_topology_generation(_dim=(10, 10), _wall_prob=0.1, _p_change_directions=0.35, _num_steps=15):
        dim_x, dim_y = _dim

        masks = [
            [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
            [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 1, 0], [1, 1, 0]],
            [[0, 0, 0], [0, 1, 1], [0, 1, 0]]
        ]

        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        direction = random.sample(directions, 1)[0]

        position = np.array([random.randint(1, dim_x - 1), random.randint(1, dim_y - 1)])

        level = np.random.choice(a=[0, 1], size=_dim, p=[_wall_prob, 1.0 - _wall_prob]).astype('int')
        # for y in range(dim_y):
        #     for x in range(dim_x):
        #         if random.random() < wall_prob:
        #             level[y][x] = 0

        for s in range(_num_steps):
            if random.random() < _p_change_directions:
                direction = random.sample(directions, 1)[0]

            position = position + direction
            position[0] = max(min(position[0], dim_x - 2), 1)
            position[1] = max(min(position[1], dim_y - 2), 1)

            mask = random.sample(masks, 1)[0]
            mask_start = position - 1
            level[mask_start[0]:mask_start[0] + 3, mask_start[1]:mask_start[1] + 3] += mask

        level[level > 0] = 1
        level[:, [0, dim_y - 1]] = 0
        level[[0, dim_x - 1], :] = 0

        return level

    def place_boxes_and_player(_room, _num_boxes):
        possible_positions = np.where(_room == 1)
        num_possible_positions = possible_positions[0].shape[0]
        num_players = 1

        if num_possible_positions <= _num_boxes + num_players:
            raise RuntimeError('Not enough free spots (#{}) to place {} player and {} boxes.'.format(num_possible_positions, num_players, _num_boxes))

        ind = np.random.randint(num_possible_positions)
        player_position = possible_positions[0][ind], possible_positions[1][ind]
        _room[player_position] = 5

        for n in range(_num_boxes):
            possible_positions = np.where(_room == 1)
            num_possible_positions = possible_positions[0].shape[0]

            ind = np.random.randint(num_possible_positions)
            box_position = possible_positions[0][ind], possible_positions[1][ind]
            _room[box_position] = 2

        return _room

    global_explored_states = set()
    global_num_boxes = 0
    global_best_room_score = -1
    global_best_room = None
    global_best_box_mapping = None

    def reverse_playing(_room_state, _room_structure):
        global global_explored_states, global_num_boxes, global_best_room_score, global_best_room, global_best_box_mapping

        box_mapping = {}
        box_locations = np.where(_room_structure == 2)
        global_num_boxes = len(box_locations[0])
        for l in range(global_num_boxes):
            box = (box_locations[0][l], box_locations[1][l])
            box_mapping[box] = box

        global_explored_states = set()
        global_best_room_score = -1
        global_best_box_mapping = box_mapping
        depth_first_search(_room_state, _room_structure, box_mapping, last_pull=(-1, -1))

        return global_best_room, global_best_room_score, global_best_box_mapping

    def depth_first_search(_room_state, _room_structure, _box_mapping, box_swaps=0, last_pull=(-1, -1), ttl=300):
        global global_explored_states, global_num_boxes, global_best_room_score, global_best_room, global_best_box_mapping

        ttl -= 1
        if ttl <= 0 or len(global_explored_states) >= 300000:
            return

        state_to_hash = marshal.dumps(_room_state)

        if not (state_to_hash in global_explored_states):
            room_score = box_swaps * box_displacement_score(_box_mapping)
            if np.where(_room_state == 2)[0].shape[0] != global_num_boxes:
                room_score = 0

            if room_score > global_best_room_score:
                global_best_room = _room_state
                global_best_room_score = room_score
                global_best_box_mapping = _box_mapping

            global_explored_states.add(state_to_hash)

            for action in ACTION_LOOKUP.keys():
                room_state_next = _room_state.copy()
                box_mapping_next = _box_mapping.copy()

                room_state_next, box_mapping_next, last_pull_next = reverse_move(room_state_next, _room_structure, box_mapping_next, last_pull, action)

                box_swaps_next = box_swaps
                if last_pull_next != last_pull:
                    box_swaps_next += 1

                depth_first_search(room_state_next, _room_structure, box_mapping_next, box_swaps_next, last_pull, ttl)

    def reverse_move(_room_state, _room_structure, _box_mapping, last_pull, action):
        player_position = np.where(_room_state == 5)
        player_position = np.array([player_position[0][0], player_position[1][0]])

        change = CHANGE_COORDINATES[action % 4]
        next_position = player_position + change

        if _room_state[next_position[0], next_position[1]] in [1, 2]:
            _room_state[player_position[0], player_position[1]] = _room_structure[player_position[0], player_position[1]]
            _room_state[next_position[0], next_position[1]] = 5

            if action < 4:
                possible_box_location = change[0] * -1, change[1] * -1
                possible_box_location += player_position

                if _room_state[possible_box_location[0], possible_box_location[1]] in [3, 4]:
                    _room_state[player_position[0], player_position[1]] = 3
                    _room_state[possible_box_location[0], possible_box_location[1]] = _room_structure[possible_box_location[0], possible_box_location[1]]

                    for k in _box_mapping.keys():
                        if _box_mapping[k] == (possible_box_location[0], possible_box_location[1]):
                            _box_mapping[k] = (player_position[0], player_position[1])
                            last_pull = k

        return _room_state, _box_mapping, last_pull

    def box_displacement_score(_box_mapping):
        score = 0

        for box_target in _box_mapping.keys():
            box_location = np.array(_box_mapping[box_target])
            box_target = np.array(box_target)
            dist = np.sum(np.abs(box_location - box_target))
            score += dist

        return score

    room_state = np.zeros(shape=dim)
    room_structure = np.zeros(shape=dim)

    score = 0
    box_mapping = None
    for t in range(tries):
        room = room_topology_generation(dim, wall_prob, p_change_directions, num_steps)
        room = place_boxes_and_player(room, _num_boxes=num_boxes)

        room_structure = np.copy(room)
        room_structure[room_structure == 5] = 1

        room_state = room.copy()
        room_state[room_state == 2] = 4

        room_state, score, box_mapping = reverse_playing(room_state, room_structure)
        room_state[room_state == 3] = 4

        if score > 0:
            break

    if score == 0:
        raise RuntimeWarning('Generated Model with score == 0')

    return room_structure, room_state, box_mapping


def create_circle_body(mass, body_type, radius):
    moment = pymunk.moment_for_circle(mass, 0.0, radius)
    body = pymunk.Body(mass, moment, body_type)
    return body


def create_circle_shape(body, radius, friction, elasticity, collision_type, sensor):
    shape = pymunk.Circle(body, radius)
    shape.friction = friction
    shape.elasticity = elasticity
    shape.collision_type = collision_type
    shape.sensor = sensor
    return shape


def create_rectangle_body(mass, body_type, half_size):
    points = [(-half_size, -half_size), (-half_size, half_size), (half_size, half_size), (half_size, -half_size)]
    moment = pymunk.moment_for_poly(mass, points)
    body = pymunk.Body(mass, moment, body_type)
    return body, points


def create_rectangle_shape(body, points, friction, elasticity, collision_type, sensor):
    shape = pymunk.Poly(body, points)
    shape.friction = friction
    shape.elasticity = elasticity
    shape.collision_type = collision_type
    shape.sensor = sensor
    return shape


def draw_circle(screen, color, position, radius):
    position = flip_y(position, screen.get_size()[1])
    pygame.draw.circle(screen, color, position.int_tuple, int(radius))


def draw_rectangle(screen, color, position, half_size):
    position = flip_y(position, screen.get_size()[1])
    points = [
        (-half_size + position.x, -half_size + position.y),
        (-half_size + position.x, half_size + position.y),
        (half_size + position.x, half_size + position.y),
        (half_size + position.x, -half_size + position.y)
    ]
    pygame.draw.polygon(screen, color, points)


def draw_sprite(screen, image, position, half_size):
    screen.blit(image, flip_y(position, screen.get_size()[1]) - Vec2d(half_size, half_size))


def flip_y(vector, y):
    return Vec2d(int(vector.x), int(-vector.y + y))


def layout_getter(layout_specs):
    def _get_layout():
        return generate_room(
            dim=(int(layout_specs['height']), int(layout_specs['width'])),
            wall_prob=float(layout_specs['wall_prob']),
            p_change_directions=float(layout_specs['p_change_directions']),
            num_steps=int(layout_specs['num_steps']),
            num_boxes=int(layout_specs['num_boxes']),
            tries=int(layout_specs['tries'])
        )[1]

    return _get_layout


def load_sprite(path, color, size=None):
    # if not existing use color, get color as parameter
    try:
        sprite = pygame.image.load(path)
        if size is not None:
            if isinstance(size, (list, tuple)):
                if len(size) != 2:
                    raise Exception
            elif isinstance(size, (float, int)):
                size = (int(size), int(size))
        sprite = pygame.transform.scale(sprite, size)
    except Exception as ex:
        print(str(ex))
        sprite = None

    if sprite is None:
        sprite = pygame.Surface((size, size))
        sprite.fill(color)

    return sprite


def load_xml(element):
    # can be common utility
    try:
        if isinstance(element, str):
            return load_xml(ElementTree.parse(element).getroot())
    except Exception as ex:
        print(str(ex))
        return {}

    d = {}
    for child in element:
        if len(child):
            d['{}'.format(child.tag)] = load_xml(child)
        else:
            d['{}'.format(child.tag)] = child.text

    return d


def euclidean_distance(position1, position2):
    return ((position1.x - position2.x) ** 2 + (position1.y - position2.y) ** 2) ** 0.5


def manhattan_distance(position1, position2):
    return abs(position1.x - position2.x) + abs(position1.y - position2.y)


def convert_to(value, type_):
    try:
        if isinstance(value, str):
            return type_(value)
        else:
            return value
    except ValueError:
        raise ValueError


class SokobanEnv(CoreEnv):
    def __init__(self, xml, xmls, sprites):
        super().__init__()

        env_specs = load_xml(xmls + xml)
        if env_specs is None:
            raise Exception

        layout_specs = env_specs.get(
            'layout', {
                'height': 7,
                'width': 7,
                'wall_prob': 0.3,
                'p_change_directions': 0.35,
                'num_steps': 5,
                'num_boxes': 1,
                'tries': 4
            }
        )
        object_specs = env_specs.get(
            'object', {
                'size': 32,
                'ground': {
                    'collision': 0,
                    'color': 'black',
                    'sprite': 'ground.png'
                },
                'goal': {
                    'collision': 1,
                    'color': 'green3',
                    'sprite': 'goal.png'
                },
                'block': {
                    'collision': 2,
                    'color': 'blue3',
                    'sprite': 'block.png'
                },
                'player': {
                    'collision': 4,
                    'color': 'orange3',
                    'sprite': 'player.png'
                },
                'obstacle': {
                    'collision': 8,
                    'color': 'red3',
                    'sprite': 'obstacle.png'
                }
            }
        )
        ground_specs = object_specs.get(
            'ground', {
                'collision': 0,
                'color': 'black',
                'sprite': 'ground.png'
            }
        )
        goal_specs = object_specs.get(
            'goal', {
                'collision': 1,
                'color': 'green3',
                'sprite': 'goal.png'
            }
        )
        block_specs = object_specs.get(
            'block', {
                'collision': 2,
                'color': 'blue3',
                'sprite': 'block.png'
            }
        )
        player_specs = object_specs.get(
            'player', {
                'collision': 4,
                'color': 'orange3',
                'sprite': 'player.png'
            }
        )
        obstacle_specs = object_specs.get(
            'obstacle', {
                'collision': 8,
                'color': 'red3',
                'sprite': 'obstacle.png'
            }
        )
        space_specs = env_specs.get(
            'space', {
                'gravity': {
                    'x': 0.0,
                    'y': 0.0
                },
                'damping': 0.0,
                'dt': 0.02,
                'steps': 2,
                'force_multiplier': 250.0
            }
        )
        gravity_specs = space_specs.get(
            'gravity', {
                'x': 0.0,
                'y': 0.0
            }
        )
        reward_specs = env_specs.get(
            'reward', {
                'movement': -0.25,
                'on_target': 5.0,
                'off_target': -5.0,
                'success': 10.0
            }
        )

        self.new_layout = layout_getter(layout_specs)
        self.layout = None

        self.obj_size = convert_to(object_specs.get('size', 32), int)
        self.world_metrics = (convert_to(layout_specs.get('width', 7), int) * self.obj_size, convert_to(layout_specs.get('height', 7), int) * self.obj_size)

        self.ground_color = THECOLORS[str(ground_specs.get('color'))]

        self.goal_collision = convert_to(goal_specs.get('collision', 1), int)
        self.goal_path = sprites + goal_specs.get('sprite')
        self.goal_color = THECOLORS[str(goal_specs.get('color'))]
        self.goal_sprite = load_sprite(path=self.goal_path, color=self.goal_color, size=self.obj_size)

        self.block_collision = convert_to(block_specs.get('collision', 2), int)
        self.block_path = sprites + block_specs.get('sprite')
        self.block_color = THECOLORS[str(block_specs.get('color'))]
        self.block_sprite = load_sprite(path=self.block_path, color=self.block_color, size=self.obj_size)

        self.player_collision = convert_to(player_specs.get('collision', 4), int)
        self.player_path = sprites + player_specs.get('sprite')
        self.player_color = THECOLORS[str(player_specs.get('color'))]
        self.player_sprite = load_sprite(path=self.player_path, color=self.player_color, size=self.obj_size)

        self.obstacle_collision = convert_to(obstacle_specs.get('collision', 8), int)
        self.obstacle_path = sprites + obstacle_specs.get('sprite')
        self.obstacle_color = THECOLORS[str(obstacle_specs.get('color'))]
        self.obstacle_sprite = load_sprite(path=self.obstacle_path, color=self.obstacle_color, size=self.obj_size)

        self.gravity = (convert_to(gravity_specs.get('x', 0.0), float), convert_to(gravity_specs.get('y', 0.0), float))
        self.damping = convert_to(space_specs.get('damping', 0.0), float)
        self.dt = convert_to(space_specs.get('dt', 0.02), float)
        self.steps = convert_to(space_specs.get('steps', 2), int)
        self.force = self.obj_size * convert_to(space_specs.get('force_multiplier', 250.0), float)

        self.movement_penalty = convert_to(reward_specs.get('movement', -0.25), float)
        self.box_on_target = convert_to(reward_specs.get('on_target', 5.0), float)
        self.box_off_target = convert_to(reward_specs.get('off_target', -5.0), float)
        self.all_boxes_on_target = convert_to(reward_specs.get('success', 10.0), float)

        self._init_space()
        self._init_window()

    def _begin(self, arbiter, _, __):
        if arbiter.shapes[0].collision_type == self.goal_collision:
            goal = arbiter.shapes[0]
            block = arbiter.shapes[1]
        else:
            goal = arbiter.shapes[1]
            block = arbiter.shapes[0]

        if (goal.body, block.body) not in self.overlaps.keys():
            self.overlaps[(goal.body, block.body)] = 0
        return True

    def _clear(self):
        self.obstacles = []
        self.players = []
        self.blocks = []
        self.goals = []

        self.player = None

        self.level_complete = False
        self.overlaps = {}
        self.space.remove(*self.space.shapes, *self.space.bodies)

    def _check_complete(self):
        self.level_complete = self.successes == len(self.goals)

    def _create_objects(self):
        for y, row in enumerate(self.layout.tolist()):
            for x, object_type in enumerate(row):
                object_position = flip_y(vector=Vec2d(self.obj_size * (x + 0.5), self.obj_size * (y + 0.5)), y=self.screen.get_size()[1])

                if object_type == 0:
                    body, points = create_rectangle_body(mass=1000000, body_type=Body.STATIC, half_size=self.obj_size * 0.5)
                    body.position = Vec2d(object_position.x, object_position.y)
                    shape = create_rectangle_shape(body=body, points=points, friction=0, elasticity=0, collision_type=self.obstacle_collision, sensor=False)
                elif object_type == 2:
                    body, points = create_rectangle_body(mass=1000000, body_type=Body.STATIC, half_size=self.obj_size * 0.5)
                    body.position = Vec2d(object_position.x, object_position.y)
                    shape = create_rectangle_shape(body=body, points=points, friction=0, elasticity=0, collision_type=self.goal_collision, sensor=True)
                elif object_type == 4:
                    body, points = create_rectangle_body(mass=1, body_type=Body.DYNAMIC, half_size=self.obj_size * 0.5)
                    body.position = Vec2d(object_position.x, object_position.y)
                    shape = create_rectangle_shape(body=body, points=points, friction=0, elasticity=0, collision_type=self.block_collision, sensor=False)
                elif object_type == 5:
                    body = create_circle_body(mass=1, body_type=Body.DYNAMIC, radius=self.obj_size * 0.5)
                    body.position = Vec2d(object_position.x, object_position.y)
                    shape = create_circle_shape(body=body, radius=self.obj_size * 0.5, friction=0, elasticity=0, collision_type=self.player_collision, sensor=False)
                else:
                    continue

                self.space.add(body)
                self.space.add(shape)

    def _draw(self):
        # clear the screen
        self.screen.fill(self.ground_color)
        self._draw_shapes()

    def _draw_shapes(self):
        for obstacle in self.obstacles:
            if self.obstacle_sprite is not None:
                draw_sprite(self.screen, image=self.obstacle_sprite, position=obstacle.body.position, half_size=self.obj_size * 0.5)
            else:
                draw_rectangle(self.screen, color=self.obstacle_color, position=obstacle.body.position, half_size=self.obj_size * 0.5)
        for goal in self.goals:
            if self.goal_sprite is not None:
                draw_sprite(self.screen, image=self.goal_sprite, position=goal.body.position, half_size=self.obj_size * 0.5)
            else:
                draw_rectangle(self.screen, color=self.goal_color, position=goal.body.position, half_size=self.obj_size * 0.5)
        for block in self.blocks:
            if self.block_sprite is not None:
                draw_sprite(self.screen, image=self.block_sprite, position=block.body.position, half_size=self.obj_size * 0.5)
            else:
                draw_rectangle(self.screen, color=self.block_color, position=block.body.position, half_size=self.obj_size * 0.5)
        for player in self.players:
            if self.player_sprite is not None:
                draw_sprite(self.screen, image=self.player_sprite, position=player.body.position, half_size=self.obj_size * 0.5)
            else:
                draw_circle(self.screen, color=self.player_color, position=player.body.position, radius=self.obj_size * 0.5)

    @staticmethod
    def _get_distance(shape, shapes, method='e'):
        min_distance = 1000000
        for other_shape in shapes:
            if method == 'e':
                distance = euclidean_distance(position1=shape.body.position, position2=other_shape.body.position)
            else:
                distance = manhattan_distance(position1=shape.body.position, position2=other_shape.body.position)

            if distance < min_distance:
                min_distance = distance

        return min_distance

    def _get_info(self):
        return self.overlaps

    def _get_observation(self):
        self._draw()
        image = pygame.surfarray.array3d(self.screen)
        image = np.swapaxes(image, 0, 1)
        return image

    def _get_reward(self):
        # remove movement penalty to agent
        total_reward = 0
        # total_reward = self.movement_penalty

        score = self._get_score()
        total_reward += (self.score - score)
        self.score = score

        # move 0.9 to xml file as well
        successes = len([v for (_, _), v in self.overlaps.items() if v >= .9])
        if successes > self.successes:
            total_reward += self.box_on_target
        elif successes < self.successes:
            total_reward += self.box_off_target
        self.successes = successes

        self._check_complete()
        if self.level_complete:
            total_reward += self.all_boxes_on_target
            return total_reward

        return total_reward

    def _get_score(self):
        total = 0
        for block in self.blocks:
            block_min = self._get_distance(block, self.goals)
            total += (block_min * 1)
        return total

    def _get_terminal(self):
        return self.level_complete

    def _init_space(self):
        self.space = pymunk.Space()
        self.space.gravity = self.gravity
        self.space.damping = self.damping

        block_goal = self.space.add_collision_handler(self.block_collision, self.goal_collision)
        block_goal.begin = self._begin
        block_goal.pre_solve = self._pre_solve
        block_goal.post_solve = self._post_solve
        block_goal.separate = self._separate

    def _init_window(self):
        pygame.init()
        self.screen = pygame.Surface(self.world_metrics)
        self.screen.fill(self.ground_color)

    def _play_step(self, force=None):
        if force is None:
            force = Vec2d(0.0, 0.0)
        self.player.force = force

        for i in range(self.steps):
            if i < self.steps - 1:
                self.space.damping = 1.0
            else:
                self.space.damping = self.damping

            for shape in self.space.shapes:
                self.space.add_post_step_callback(self._post_step_callback, shape)
            self.space.step(self.dt)

    def _post_solve(self, arbiter, _, __):
        if arbiter.shapes[0].collision_type == self.goal_collision:
            goal = arbiter.shapes[0]
            block = arbiter.shapes[1]
        else:
            goal = arbiter.shapes[1]
            block = arbiter.shapes[0]

        goal_center = goal.body.position
        block_center = block.body.position

        center_difference = Vec2d(abs(goal_center.x - block_center.x), abs(goal_center.y - block_center.y))
        filled_area_sizes = Vec2d(self.obj_size, self.obj_size) - center_difference
        filled_area = abs(filled_area_sizes.x * filled_area_sizes.y)
        filled_area_percentage = filled_area / goal.area

        self.overlaps[(goal.body, block.body)] = filled_area_percentage

    @staticmethod
    def _post_step_callback(space, key):
        key.body.angle = 0
        space.reindex_shapes_for_body(key.body)

    def _pre_solve(self, arbiter, _, __):
        if arbiter.shapes[0].collision_type == self.goal_collision:
            goal = arbiter.shapes[0]
            block = arbiter.shapes[1]
        else:
            goal = arbiter.shapes[1]
            block = arbiter.shapes[0]

        goal_center = goal.body.position
        block_center = block.body.position

        center_difference = Vec2d(abs(goal_center.x - block_center.x), abs(goal_center.y - block_center.y))
        filled_area_sizes = Vec2d(self.obj_size, self.obj_size) - center_difference
        filled_area = abs(filled_area_sizes.x * filled_area_sizes.y)
        filled_area_percentage = filled_area / goal.area

        self.overlaps[(goal.body, block.body)] = filled_area_percentage
        return True

    def _recreate(self):
        self._create_objects()

        self.obstacles = list(filter(lambda shape: shape.collision_type == self.obstacle_collision, self.space.shapes))
        self.players = list(filter(lambda shape: shape.collision_type == self.player_collision, self.space.shapes))
        self.blocks = list(filter(lambda shape: shape.collision_type == self.block_collision, self.space.shapes))
        self.goals = list(filter(lambda shape: shape.collision_type == self.goal_collision, self.space.shapes))

        self.player = self.players[0].body

        self.score = self._get_score()
        self.successes = 0

    def _separate(self, arbiter, _, __):
        if arbiter.shapes[0].collision_type == self.goal_collision:
            goal = arbiter.shapes[0]
            block = arbiter.shapes[1]
        else:
            goal = arbiter.shapes[1]
            block = arbiter.shapes[0]

        if (goal.body, block.body) in self.overlaps.keys():
            del self.overlaps[(goal.body, block.body)]

    def close(self):
        self._clear()
        cv2.destroyAllWindows()
        pygame.quit()

    def render(self, mode='human'):
        image = pygame.surfarray.array3d(self.screen)
        image = np.swapaxes(image, 0, 1)
        cv2.imshow('state', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def reset(self):
        self._clear()
        self.layout = self.new_layout()
        self._recreate()
        self._play_step()
        return self._get_observation()

    def seed(self, seed=None):
        pass

    def step(self, action):
        try:
            x, y = action
            vector = Vec2d(x, y)
            if vector.length > 1.0:
                vector = vector.normalized
        except TypeError:
            vector = Vec2d(0.0, 0.0)
        self._play_step(force=vector * self.force)
        return self._get_observation(), self._get_reward(), self._get_terminal(), self._get_info()

    def goal(self):
        screen = pygame.Surface(self.world_metrics)
        screen.fill(self.ground_color)

        for obstacle in self.obstacles:
            if self.obstacle_sprite is not None:
                draw_sprite(
                    screen,
                    image=self.obstacle_sprite,
                    position=obstacle.body.position,
                    half_size=self.obj_size * 0.5
                )
            else:
                draw_rectangle(
                    screen,
                    color=self.obstacle_color,
                    position=obstacle.body.position,
                    half_size=self.obj_size * 0.5
                )

        player = self.players[0]
        for goal in self.goals:
            if self.goal_sprite is not None:
                draw_sprite(
                    screen,
                    image=self.goal_sprite,
                    position=goal.body.position,
                    half_size=self.obj_size * 0.5
                )
            else:
                draw_rectangle(
                    screen,
                    color=self.goal_color,
                    position=goal.body.position,
                    half_size=self.obj_size * 0.5
                )

            if self.block_sprite is not None:
                draw_sprite(
                    screen,
                    image=self.block_sprite,
                    position=player.body.position - (player.body.position - goal.body.position) * 0.95,
                    half_size=self.obj_size * 0.5
                )
            else:
                draw_rectangle(
                    screen,
                    color=self.block_color,
                    position=player.body.position - (player.body.position - goal.body.position) * 0.95,
                    half_size=self.obj_size * 0.5
                )

        player = self.players[0]
        block = self.blocks[0]
        if self.player_sprite is not None:
            draw_sprite(
                screen, image=self.player_sprite, position=player.body.position - (player.body.position - block.body.position) * 0.9,
                half_size=self.obj_size * 0.5
            )
        else:
            draw_circle(
                screen, color=self.player_color, position=player.body.position - (player.body.position - block.body.position) * 0.9,
                radius=self.obj_size * 0.5
            )

        image = pygame.surfarray.array3d(screen)
        image = np.swapaxes(image, 0, 1)
        return image
