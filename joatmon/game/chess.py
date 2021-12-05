import dataclasses
import math

import pygame
from chess import (
    BB_LIGHT_SQUARES,
    BB_SQUARES,
    BISHOP,
    Board,
    FILE_NAMES,
    KNIGHT,
    Move,
    PAWN,
    QUEEN,
    RANK_NAMES,
    ROOK,
    Square,
    WHITE
)
from pygame.color import THECOLORS

__all__ = ['ChessEnv', 'create_move_labels']


@dataclasses.dataclass(unsafe_hash=True)
class Arrow:
    from_square: Square
    to_square: Square
    color: str = None

    def __eq__(self, other):
        return self.from_square == other.from_square and self.to_square == other.to_square and self.color == other.color


def arrow(screen, color, start, end, thickness):
    # arrow body
    pygame.draw.line(
        screen,
        color,
        start,
        end,
        thickness
    )
    rotation = (math.atan2(start[1] - end[1], end[0] - start[0])) + math.pi / 2
    # arrow head
    pygame.draw.polygon(
        screen,
        color,
        (
            (end[0] + 9 * math.sin(rotation), end[1] + 9 * math.cos(rotation)),
            (end[0] + 9 * math.sin(rotation - 120 * math.pi / 180), end[1] + 9 * math.cos(rotation - 120 * math.pi / 180)),
            (end[0] + 9 * math.sin(rotation + 120 * math.pi / 180), end[1] + 9 * math.cos(rotation + 120 * math.pi / 180))
        )
    )


setattr(pygame.draw, 'arrow', arrow)

pygame.init()

SQUARE_SIZE = 64
MARGIN = 20
empty_space = 40
light_border_size = 5
dark_border_size = 3
piece_size = 64

PIECES = {
    "b": pygame.transform.scale(pygame.image.load('joatmon/game/assets/chess/black-bishop.png'), (SQUARE_SIZE, SQUARE_SIZE)),
    "k": pygame.transform.scale(pygame.image.load('joatmon/game/assets/chess/black-king.png'), (SQUARE_SIZE, SQUARE_SIZE)),
    "n": pygame.transform.scale(pygame.image.load('joatmon/game/assets/chess/black-knight.png'), (SQUARE_SIZE, SQUARE_SIZE)),
    "p": pygame.transform.scale(pygame.image.load('joatmon/game/assets/chess/black-pawn.png'), (SQUARE_SIZE, SQUARE_SIZE)),
    "q": pygame.transform.scale(pygame.image.load('joatmon/game/assets/chess/black-queen.png'), (SQUARE_SIZE, SQUARE_SIZE)),
    "r": pygame.transform.scale(pygame.image.load('joatmon/game/assets/chess/black-rook.png'), (SQUARE_SIZE, SQUARE_SIZE)),
    "B": pygame.transform.scale(pygame.image.load('joatmon/game/assets/chess/white-bishop.png'), (SQUARE_SIZE, SQUARE_SIZE)),
    "K": pygame.transform.scale(pygame.image.load('joatmon/game/assets/chess/white-king.png'), (SQUARE_SIZE, SQUARE_SIZE)),
    "N": pygame.transform.scale(pygame.image.load('joatmon/game/assets/chess/white-knight.png'), (SQUARE_SIZE, SQUARE_SIZE)),
    "P": pygame.transform.scale(pygame.image.load('joatmon/game/assets/chess/white-pawn.png'), (SQUARE_SIZE, SQUARE_SIZE)),
    "Q": pygame.transform.scale(pygame.image.load('joatmon/game/assets/chess/white-queen.png'), (SQUARE_SIZE, SQUARE_SIZE)),
    "R": pygame.transform.scale(pygame.image.load('joatmon/game/assets/chess/white-rook.png'), (SQUARE_SIZE, SQUARE_SIZE))
}


def create_move_labels():
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    promoted_to = ['q', 'r', 'b', 'n']

    for l1 in range(8):
        for n1 in range(8):
            destinations = [(t, n1) for t in range(8)] +\
                           [(l1, t) for t in range(8)] +\
                           [(l1 + t, n1 + t) for t in range(-7, 8)] +\
                           [(l1 + t, n1 - t) for t in range(-7, 8)] +\
                           [(l1 + a, n1 + b) for (a, b) in [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(8) and n2 in range(8):
                    move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                    labels_array.append(move)
    for l1 in range(8):
        letter = letters[l1]
        for p in promoted_to:
            labels_array.append(letter + '2' + letter + '1' + p)
            labels_array.append(letter + '7' + letter + '8' + p)
            if l1 > 0:
                l_l = letters[l1 - 1]
                labels_array.append(letter + '2' + l_l + '1' + p)
                labels_array.append(letter + '7' + l_l + '8' + p)
            if l1 < 7:
                l_r = letters[l1 + 1]
                labels_array.append(letter + '2' + l_r + '1' + p)
                labels_array.append(letter + '7' + l_r + '8' + p)
    return labels_array


class ChessEnv:
    def __init__(self):
        self.board = Board()
        self.board_size = empty_space + light_border_size + dark_border_size + piece_size * 8 + dark_border_size + light_border_size + empty_space
        self.screen = pygame.display.set_mode((self.board_size, self.board_size))

        self.reset()

    def __eq__(self, env):
        if isinstance(env, ChessEnv):
            return self.board == env.board
        else:
            return NotImplemented

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        env = self.copy()
        memo[id(self)] = env
        return env

    def copy(self, *, stack=True):
        env = type(self)()
        env.board = env.board.copy(stack=stack)
        return env

    def reset(self):
        self.board.reset()
        return self.get_observation()

    def get_observation(self, orientation=WHITE, flipped=False, mode='str'):
        if mode == 'rgb':
            orientation ^= flipped

            surface = pygame.Surface((piece_size * 8, piece_size * 8))

            surface.fill(THECOLORS['black'])

            for square, bb in enumerate(BB_SQUARES):
                file_index = square & 7
                rank_index = square >> 3

                x = (file_index if orientation else 7 - file_index) * piece_size
                y = (7 - rank_index if orientation else rank_index) * piece_size

                color = "white" if BB_LIGHT_SQUARES & bb else "grey"

                pygame.draw.polygon(
                    surface,
                    THECOLORS[color],
                    [
                        (x, y),
                        (x, y + piece_size),
                        (x + piece_size, y + piece_size),
                        (x + piece_size, y)
                    ]
                )

                _piece = self.board.piece_at(square)

                if _piece is not None:
                    surface.blit(
                        PIECES[_piece.symbol()],
                        (
                            x,
                            y
                        )
                    )

            return surface
        else:
            state = []
            for square, bb in enumerate(BB_SQUARES):
                _piece = self.board.piece_at(square)
                if _piece is not None:
                    state.append(_piece.piece_type * (1 if _piece.color else -1))
                else:
                    state.append(0)
            return state

    def get_reward(self, piece):
        reward = 0
        if piece == PAWN:
            reward += 1
        if piece == KNIGHT:
            reward += 3.05
        if piece == BISHOP:
            reward += 3.33
        if piece == ROOK:
            reward += 5.63
        if piece == QUEEN:
            reward += 9.5
        if self.board.is_checkmate():
            reward += 100
        if self.board.turn:
            reward *= -1
        return reward

    def step(self, move, move_type='uci'):
        if move_type == 'san':
            move = self.board.parse_san(self.board.san(move))
        captured_piece_type = self.board.piece_at(move.to_square)
        return self.get_observation(), self.get_reward(captured_piece_type), self.board.is_game_over(), {'turn': not self.board.turn}

    def result(self, *, claim_draw=False):
        if self.board.is_checkmate():
            return "0-1" if self.board.turn == WHITE else "1-0"

        if claim_draw and self.board.can_claim_draw():
            return "1/2-1/2"

        if self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
            return "1/2-1/2"

        if self.board.is_insufficient_material():
            return "1/2-1/2"

        if not any(self.board.generate_legal_moves()):
            return "1/2-1/2"

        return "*"

    def unicode(self, *, invert_color=False, borders=False, empty_square="⭘"):
        builder = []
        for rank_index in range(7, -1, -1):
            if borders:
                builder.append("  ")
                builder.append("-" * 17)
                builder.append("\n")

                builder.append(RANK_NAMES[rank_index])
                builder.append(" ")

            for file_index in range(8):
                square_index = rank_index * 8 + file_index

                if borders:
                    builder.append("|")
                elif file_index > 0:
                    builder.append(" ")

                _piece = self.board.piece_at(square_index)

                if _piece:
                    builder.append(_piece.unicode_symbol(invert_color=invert_color))
                else:
                    builder.append(empty_square)

            if borders:
                builder.append("|")

            if borders or rank_index > 0:
                builder.append("\n")

        if borders:
            builder.append("  ")
            builder.append("-" * 17)
            builder.append("\n")
            builder.append("   a b c d e f g h")

        return "".join(builder)

    def render(self, *, orientation=WHITE, flipped=False, mode='str'):
        if mode == 'rgb':
            orientation ^= flipped

            self.screen.fill(THECOLORS['grey'])

            font = pygame.font.SysFont(None, 36)

            for idx, file in enumerate(FILE_NAMES):
                img = font.render(file, True, THECOLORS['white'])
                self.screen.blit(
                    img,
                    (
                        empty_space + light_border_size + dark_border_size + piece_size // 2 + idx * piece_size,
                        empty_space + light_border_size + dark_border_size + piece_size // 4 + 8 * piece_size
                    )
                )

            for idx, rank in enumerate(RANK_NAMES):
                img = font.render(rank, True, THECOLORS['white'])
                self.screen.blit(
                    img,
                    (
                        empty_space // 2,
                        self.board_size + piece_size // 4 - empty_space - light_border_size - dark_border_size - (idx + 1) * piece_size
                    )
                )

            pygame.draw.polygon(
                self.screen,
                THECOLORS['white'],
                [
                    (empty_space, empty_space),
                    (empty_space, -empty_space + self.board_size),
                    (-empty_space + self.board_size, -empty_space + self.board_size),
                    (-empty_space + self.board_size, empty_space)
                ]
            )

            pygame.draw.polygon(
                self.screen,
                THECOLORS['black'],
                [
                    (empty_space + light_border_size, empty_space + light_border_size),
                    (empty_space + light_border_size, -empty_space - light_border_size + self.board_size),
                    (-empty_space - light_border_size + self.board_size, -empty_space - light_border_size + self.board_size),
                    (-empty_space - light_border_size + self.board_size, empty_space + light_border_size)
                ]
            )

            last_move = self.board.peek() or Move(from_square=-1, to_square=-1)
            arrows = set()
            arrows.add(Arrow(from_square=last_move.from_square, to_square=last_move.to_square, color='green'))

            for square, bb in enumerate(BB_SQUARES):
                file_index = square & 7
                rank_index = square >> 3

                x = (file_index if orientation else 7 - file_index) * piece_size + empty_space + light_border_size + dark_border_size
                y = (7 - rank_index if orientation else rank_index) * piece_size + empty_space + light_border_size + dark_border_size

                color = "white" if BB_LIGHT_SQUARES & bb else "grey"

                if self.board.piece_at(square) is not None and self.board.piece_at(square).color == self.board.turn:
                    for attacker in self.board.attackers(not self.board.turn, square):
                        if self.board.piece_at(attacker) is not None and self.board.piece_at(attacker).color != self.board.turn:
                            arrows.add(Arrow(from_square=attacker, to_square=square, color='red'))

                if self.board.piece_at(square) is not None and self.board.piece_at(square).color == self.board.turn:
                    for attack in self.board.attacks(square):
                        if self.board.piece_at(attack) is not None and self.board.piece_at(attack).color != self.board.turn:
                            arrows.add(Arrow(from_square=square, to_square=attack, color='orange'))

                pygame.draw.polygon(
                    self.screen,
                    THECOLORS[color],
                    [
                        (x, y),
                        (x, y + piece_size),
                        (x + piece_size, y + piece_size),
                        (x + piece_size, y)
                    ]
                )

                _piece = self.board.piece_at(square)

                if _piece is not None:
                    self.screen.blit(
                        PIECES[_piece.symbol()],
                        (
                            x,
                            y
                        )
                    )

            for _arrow in arrows:
                from_file_index = _arrow.from_square & 7
                from_rank_index = _arrow.from_square >> 3

                to_file_index = _arrow.to_square & 7
                to_rank_index = _arrow.to_square >> 3

                from_x = (from_file_index if orientation else 7 - from_file_index) * piece_size + empty_space + light_border_size + dark_border_size
                from_y = (7 - from_rank_index if orientation else from_rank_index) * piece_size + empty_space + light_border_size + dark_border_size

                to_x = (to_file_index if orientation else 7 - to_file_index) * piece_size + empty_space + light_border_size + dark_border_size
                to_y = (7 - to_rank_index if orientation else to_rank_index) * piece_size + empty_space + light_border_size + dark_border_size

                pygame.draw.arrow(
                    self.screen,
                    THECOLORS[_arrow.color],
                    (from_x + piece_size // 2, from_y + piece_size // 2),
                    (to_x + piece_size // 2, to_y + piece_size // 2),
                    5
                )

            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit(1)
        else:
            print(self.unicode())
