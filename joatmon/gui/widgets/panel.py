import pygame

from joatmon.event import Event


class Panel:
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
            'change': Event()
        }

        self.is_hovered = False
        self.is_clicked = False
        self.is_active = False

        self._placeholder = ''
        self._text = ''

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

    def placeholder(self, placeholder):
        self._placeholder = placeholder

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self._rect.collidepoint(event.pos)
            if self.is_hovered:
                _event = self._events.get('hover', None)
                if _event:
                    _event.fire()

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self._rect.collidepoint(event.pos):
                self.is_active = True
            else:
                self.is_active = False
        elif event.type == pygame.KEYDOWN:
            if self.is_active:
                prev_text = self._text

                if event.key == pygame.K_RETURN:
                    self._text = ''
                elif event.key == pygame.K_BACKSPACE:
                    self._text = self._text[:-1]
                else:
                    self._text += event.unicode

                if prev_text != self._text:
                    _event = self._events.get('change', None)
                    if _event:
                        _event.fire()

    def draw(self, screen):
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

                pygame.draw.rect(screen, pygame.Color('gray') if self.is_hovered else pygame.Color('white'), border_rect, border_radius=radius)
            pygame.draw.rect(screen, border_color, border_rect, thickness, border_radius=radius)

        font = pygame.font.SysFont(None, 24)

        # pygame.draw.rect(screen, self._background_color, self._rect, 0)
        # pygame.draw.rect(screen, self._border.get('color', (0, 0, 0)), self._rect, 2)

        text_surface = font.render(self._text if self._text is not None and self._text != '' else self._placeholder, True, self._text_color)
        screen.blit(text_surface, (self._rect.x + 5, self._rect.y + 5))
