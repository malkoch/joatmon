import pygame

from joatmon.event import Event
from joatmon.system.gui.widgets.label import Label
from joatmon.system.gui.widgets.panel import Panel
from joatmon.system.gui.widgets.widget import Element


class TabView(Element):
    def __init__(self, id_):
        super(TabView, self).__init__(id_)

        self._content = Panel(f'{id_}.content').layout('vertical')
        self._header = Panel(f'{id_}.header').layout('horizontal').style({'height': 1, 'border': {'color': (0, 0, 0), 'thickness': 1, 'shadow': 1, 'radius': 1}})
        self._body = Panel(f'{id_}.body').layout('vertical').style({'height': 9, 'border': {'color': (0, 0, 0), 'thickness': 1, 'shadow': 1, 'radius': 1}})

        self._content.add(self._header)
        self._content.add(self._body)

        self._bodies = {}

        self.add(self._content)

        self._selected_tab = None
        self._events['tab_change'] = Event()

    def _tab_changed(self, tab):
        self._events.get('tab_change').fire(tab)
        self._selected_tab = tab
        self._body._elements = self._bodies[self._selected_tab._id]._elements

        self.update()

    def new_tab(self, tab):
        t = Label(f'{tab}').text(tab).style({'border': {'color': (0, 0, 0), 'thickness': 1, 'shadow': 1, 'radius': 1}}).on('click', self._tab_changed)
        self._header.add(t)

        body = Panel('').layout('vertical').style({'height': 9, 'border': {'color': (0, 0, 0), 'thickness': 1, 'shadow': 1, 'radius': 1}})

        self._bodies[tab] = body

        if self._selected_tab is None:
            self._tab_changed(t)

        return self

    def update_tab(self, tab, *content):
        if tab not in self._bodies:
            self.new_tab(tab)

        body = self._bodies[tab]
        body._elements = []

        for row in content:
            self.new_row(tab, row)

        if self._selected_tab._id == tab:
            self._body._elements = body._elements

        return self

    def get_tab(self, tab):
        body = self._bodies[tab]
        return body

    def new_row(self, tab, text):
        body = self._bodies[tab]
        body.add(Label('').text(text).style({'border': {'color': (0, 0, 0), 'thickness': 1, 'shadow': 1, 'radius': 1}}))
        self._bodies[tab] = body

        return self

    def draw(self, screen):
        self.update()

        super(TabView, self).draw(screen)

        if self._selected_tab:
            pygame.draw.rect(screen, (222, 222, 222), self._selected_tab._rect)
