import pygame
from pygame.locals import MOUSEBUTTONDOWN

from joatmon.core.event import Event

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)


class ComboBox:
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self):
        self._x = None
        self._y = None

        self._width = None
        self._height = None

        self._text = None
        self._font = None

        self._text_color = None
        self._background_color = None
        self._hover_color = None
        self._click_color = None

        self._border = None
        self._rect = None
        self._icon = None

        self._events = {'hover': Event(), 'change': Event()}

        self._options = []

        self.is_hovered = False
        self.is_clicked = False

        self.is_open = False
        self.selected_option = None

    def position(self, x, y):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self._x = x
        self._y = y

        return self

    def size(self, width, height):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self._width = width
        self._height = height

        self._rect = pygame.Rect(self._x, self._y, self._width, self._height)

        return self

    def icon(self, icon):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        _icon = pygame.image.load(icon)

        self._width = self._height
        # self._rect = None
        self._rect = pygame.Rect(self._x, self._y, self._width, self._height)

        self._icon = pygame.transform.scale(_icon, (self._width - 10, self._height - 10))

        return self

    def text(self, text, font):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self._text = text
        self._font = font

        return self

    def color(self, t_color, b_color, h_color, c_color):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self._text_color = t_color
        self._background_color = b_color
        self._hover_color = h_color
        self._click_color = c_color

        return self

    def border(self, border):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self._border = border
        return self

    def on(self, event, func):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        _event = self._events.get(event, None)
        if _event:
            _event += func

        return self

    def add_option(self, option):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self._options.append(option)
        return self

    def remove_option(self, option):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return self

    def handle_event(self, event):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self._rect.collidepoint(event.pos)
            if self.is_hovered:
                _event = self._events.get('hover', None)
                if _event:
                    _event.fire()

        if event.type == MOUSEBUTTONDOWN:
            if self._rect.collidepoint(event.pos):
                self.is_open = not self.is_open
            else:
                self.is_open = False

            # if event.type == MOUSEBUTTONUP and self.is_open:
            #     print('mouse up')

            for i, option_rect in enumerate(self.option_rects):
                if self.is_open and option_rect.collidepoint(event.pos):
                    self.selected_option = self._options[i]
                    self.is_open = False

                    _event = self._events.get('change', None)
                    if _event:
                        _event.fire()

                    break

        self.update()

    def update(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.option_rects = []

        for i, option in enumerate(self._options):
            option_rect = pygame.Rect(
                self._rect.x, self._rect.y + self._rect.height * (i + 1), self._rect.width, self._rect.height
            )
            self.option_rects.append(option_rect)

    def draw(self, screen):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        border_color = self._border.get('color', (0, 0, 0))
        thickness = self._border.get('thickness', 1)
        radius = self._border.get('radius', 1)
        shadow = self._border.get('shadow', 3)

        if self._border:
            border_rect = self._rect.copy()
            border_rect.x -= thickness
            border_rect.y -= thickness
            border_rect.width += 2 * thickness
            border_rect.height += 2 * thickness

            if shadow:
                shadow_rect = border_rect.copy()
                shadow_rect.x += shadow
                shadow_rect.y += shadow
                pygame.draw.rect(screen, pygame.Color('gray'), shadow_rect, border_radius=radius)  # shadow color

                pygame.draw.rect(
                    screen,
                    pygame.Color('gray') if self.is_hovered else pygame.Color('white'),
                    border_rect,
                    border_radius=radius,
                )
            pygame.draw.rect(screen, border_color, border_rect, thickness, border_radius=radius)

        font = pygame.font.SysFont(None, 24)

        # pygame.draw.rect(screen, WHITE, self._rect)
        # pygame.draw.rect(screen, BLACK, self._rect, 2)
        #
        # pygame.draw.line(
        #    screen, BLACK, (self._rect.x + self._rect.width - 20, self._rect.y),
        #    (self._rect.x + self._rect.width - 20, self._rect.y + self._rect.height), 2
        # )

        if self.is_open:
            for i, option_rect in enumerate(self.option_rects):  # if this is the selected, change the color as well
                pygame.draw.rect(screen, GRAY, option_rect)
                pygame.draw.rect(screen, BLACK, option_rect, 1)
                text_surface = font.render(self._options[i], True, BLACK)
                screen.blit(text_surface, (option_rect.x + 5, option_rect.y + 5))

        if self.selected_option:
            text_surface = font.render(self.selected_option, True, BLACK)
            screen.blit(text_surface, (self._rect.x + 5, self._rect.y + 5))
