import pygame

from joatmon.core.event import Event


class Element:
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

    def __init__(self, id_):
        self._id = id_

        self._original_w = 1
        self._original_h = 1

        self._x = 0
        self._y = 0

        self._width = 1
        self._height = 1

        self._color = (255, 255, 255)

        self._border = {'color': (0, 0, 0), 'thickness': 1, 'shadow': 1, 'radius': 1}
        self._rect = None

        self._surface = None
        self._elements = []
        self._layout = 'vertical'

        self._events = {'click': Event()}

    def layout(self, layout):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self._layout = layout

        return self

    def add(self, element):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self._elements.append(element)
        self.update()

        return self

    def style(self, style):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self._width = style.get('width', self._width)
        self._height = style.get('height', self._height)

        self._original_w = style.get('width', self._original_w)
        self._original_h = style.get('height', self._original_h)

        self._rect = pygame.Rect(self._x, self._y, self._width, self._height)
        # self._surface = pygame.Surface((self._width, self._height))

        self._color = style.get('color', self._color)

        self._border = style.get('border', self._border)

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

    def update(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self._rect = pygame.Rect(self._x, self._y, self._width, self._height)

        if self._layout == 'vertical':
            h_weights = list(map(lambda x: x._original_h, self._elements))
            w_weights = list(map(lambda x: 1, self._elements))
        else:
            h_weights = list(map(lambda x: 1, self._elements))
            w_weights = list(map(lambda x: x._original_w, self._elements))

        for idx, element in enumerate(self._elements):
            if self._layout == 'vertical':
                element._width = self._width
                element._height = h_weights[idx] * (self._height / sum(h_weights))
            else:
                element._width = w_weights[idx] * (self._width / sum(w_weights))
                element._height = self._height

        current_x = self._x
        current_y = self._y
        for element in self._elements:
            element._x = current_x
            element._y = current_y

            if self._layout == 'vertical':
                current_y += element._height
            else:
                current_x += element._width

        for element in self._elements:
            element.update()

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

                pygame.draw.rect(screen, self._color, border_rect, border_radius=radius)
            pygame.draw.rect(screen, border_color, border_rect, thickness, border_radius=radius)

        pygame.draw.rect(screen, self._color, self._rect)

        for element in self._elements:
            element.draw(screen)

    def handle(self, event):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for element in self._elements:
            element.handle(event)

        if event.type == pygame.MOUSEBUTTONDOWN and self._rect.collidepoint(event.pos):
            _event = self._events.get('click', None)
            if _event:
                _event.fire(self)
