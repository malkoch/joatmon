import pygame

from joatmon.system.gui.widgets.widget import Element


class Button(Element):
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
        super(Button, self).__init__(id_)

        self._text = None
        self._font = pygame.font.SysFont(None, 24)

        self._text_color = (0, 0, 0)
        self._background_color = (255, 255, 255)
        self._hover_color = (127, 127, 127)
        self._click_color = (63, 63, 63)

        self._icon = None

        self.is_hovered = False
        self.is_clicked = False

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

    def text(self, text):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self._text = text

        return self

    def style(self, style):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        super(Button, self).style(style)

        self._text_color = style.get('text_color', self._text_color)
        self._hover_color = style.get('hover_color', self._hover_color)
        self._click_color = style.get('click_color', self._click_color)

        return self

    def draw(self, screen):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        color = (
            self._click_color if self.is_clicked else self._hover_color if self.is_hovered else self._background_color
        )

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

                pygame.draw.rect(screen, color, border_rect, border_radius=radius)
            pygame.draw.rect(screen, border_color, border_rect, thickness, border_radius=radius)

        if self._icon:
            icon_rect = self._icon.get_rect(center=self._rect.center)
            screen.blit(self._icon, icon_rect)
        else:
            pygame.draw.rect(screen, color, self._rect)

            text_surface = self._font.render(self._text, True, self._text_color)
            text_rect = text_surface.get_rect(center=self._rect.center)
            screen.blit(text_surface, text_rect)

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
        elif event.type == pygame.MOUSEBUTTONDOWN and self.is_hovered:
            self.is_clicked = True
            _event = self._events.get('click', None)
            if _event:
                _event.fire()
        elif event.type == pygame.MOUSEBUTTONUP:
            self.is_clicked = False
