import pygame

from joatmon.system.gui.widgets.widget import Element


class Label(Element):
    def __init__(self, id_):
        super(Label, self).__init__(id_)

        self._text = None
        self._text_color = (0, 0, 0)
        self._font = pygame.font.SysFont(None, 24)

    def text(self, text):
        self._text = text

        return self

    def style(self, style):
        super(Label, self).style(style)

        self._text_color = style.get('text_color', self._text_color)
        self._font = style.get('font', self._font)

        return self

    def draw(self, screen):
        super(Label, self).draw(screen)

        text_surface = self._font.render(self._text, True, self._text_color)
        text_rect = text_surface.get_rect(center=self._rect.center)
        screen.blit(text_surface, text_rect)
