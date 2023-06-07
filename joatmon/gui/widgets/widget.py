import pygame

from joatmon.event import Event


class Element:
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

        self._events = {
            'hover': Event(),
            'click': Event()
        }

        self.is_hovered = False
        self.is_clicked = False

    def position(self, x, y):
        self._x = x
        self._y = y

        return self

    def size(self, width, height):
        self._width = width
        self._height = height

        self._rect = pygame.Rect(self._x, self._y, self._width, self._height)

        return self

    def icon(self, icon):
        _icon = pygame.image.load(icon)

        self._width = self._height
        # self._rect = None
        self._rect = pygame.Rect(self._x, self._y, self._width, self._height)

        self._icon = pygame.transform.scale(_icon, (self._width - 10, self._height - 10))

        return self

    def text(self, text, font):
        self._text = text
        self._font = font

        return self

    def color(self, t_color, b_color, h_color, c_color):
        self._text_color = t_color
        self._background_color = b_color
        self._hover_color = h_color
        self._click_color = c_color

        return self

    def border(self, border):
        self._border = border
        return self

    def on(self, event, func):
        _event = self._events.get(event, None)
        if _event:
            _event += func

        return self

    def draw(self, screen):
        color = self._click_color if self.is_clicked else self._hover_color if self.is_hovered else self._background_color

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
