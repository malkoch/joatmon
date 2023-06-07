import pygame

from joatmon.gui.widgets.button import Button
from joatmon.gui.widgets.dropdown import ComboBox
from joatmon.gui.widgets.lineedit import InputField


def button_clicked():
    print("Button clicked!")


def combobox_changed():
    print("Combobox changed!")


def lineedit_changed():
    print("Lineedit changed!")


pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

button_size = (100, 30)
combobox_size = (100, 25)
lineedit_size = (100, 25)
base_position = (300, 200)

elements = []

for y in range(3):
    for x in range(3):
        if y == 0:
            element = Button().position(
                base_position[0] + button_size[0] * x + 10 * x,
                base_position[1]
            ).size(*button_size).text('Click Me', font).color(
                t_color=(0, 0, 0),
                b_color=(255, 255, 255),
                c_color=(63, 63, 63),
                h_color=(127, 127, 127)
            ).border({'color': (0, 0, 0), 'thickness': 3, 'shadow': 3, 'radius': 3}).on('click', button_clicked)
            if x == 0:
                element = element.icon(r'C:\Users\malkoch\Documents\GitHub\joatmon\joatmon\assistant\assets\add.png')
            if x == 1:
                element = element.icon(r'C:\Users\malkoch\Documents\GitHub\joatmon\joatmon\assistant\assets\run.png')
            if x == 2:
                element = element.icon(r'C:\Users\malkoch\Documents\GitHub\joatmon\joatmon\assistant\assets\setting.png')
            elements.append(element)
        if y == 1:
            element = ComboBox().position(
                base_position[0] + combobox_size[0] * x + 10 * x,
                base_position[1] + button_size[1] + 10
            ).size(*combobox_size).text('Click Me', font).color(
                t_color=(0, 0, 0),
                b_color=(255, 255, 255),
                c_color=(63, 63, 63),
                h_color=(127, 127, 127)
            ).border({'color': (0, 0, 0), 'thickness': 3, 'shadow': 3, 'radius': 3}).on('change', combobox_changed)
            if x == 0:
                element = element.add_option('123')
                element = element.add_option('123')
            if x == 1:
                element = element.add_option('234')
                element = element.add_option('234')
            if x == 2:
                element = element.add_option('345')
                element = element.add_option('345')
            elements.append(element)
        if y == 2:
            element = InputField().position(
                base_position[0] + lineedit_size[0] * x + 10 * x,
                base_position[1] + button_size[1] + 10 + combobox_size[1] + 10
            ).size(*lineedit_size).text('Click Me', font).color(
                t_color=(0, 0, 0),
                b_color=(255, 255, 255),
                c_color=(63, 63, 63),
                h_color=(127, 127, 127)
            ).border({'color': (0, 0, 0), 'thickness': 3, 'shadow': 3, 'radius': 3}).on('change', lineedit_changed)
            if x == 0:
                element.placeholder('123')
            if x == 1:
                element.placeholder('234')
            if x == 2:
                element.placeholder('345')
            elements.append(element)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        else:
            for element in elements:
                element.handle_event(event)

    screen.fill((255, 255, 255))
    for element in elements:
        element.draw(screen)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
