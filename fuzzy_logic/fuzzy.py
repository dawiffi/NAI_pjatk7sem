import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from skfuzzy.control.visualization import FuzzyVariableVisualizer
import pygame_widgets
import pygame
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox

## notatki
# na jutro: https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem_newapi.html
# + plus poprawić początek na ten z aktualnego tutoriala

## Fuzzification

# Generate universe variables
temp = ctrl.Antecedent(np.arange(30, 90, 0.1), "temp")  # in C
load = ctrl.Antecedent(np.arange(0, 101, 1), "load")  # in %
tdp = ctrl.Antecedent(np.arange(0, 250, 1), "tdp")  # in Watts

fan_speed = ctrl.Consequent(np.arange(0, 101, 1), "fan_speed")  # (out) %

# Generate fuzzy membership functions
temp["low"] = fuzz.trimf(temp.universe, [30, 30, 50])
temp["medium"] = fuzz.trimf(temp.universe, [40, 60, 80])
temp["high"] = fuzz.trimf(temp.universe, [70, 90, 90])

load["low"] = fuzz.trimf(load.universe, [0, 0, 30])
load["medium"] = fuzz.trimf(load.universe, [20, 50, 80])
load["high"] = fuzz.trimf(load.universe, [70, 100, 100])

tdp["low"] = fuzz.trimf(tdp.universe, [0, 0, 40])
tdp["medium"] = fuzz.trimf(tdp.universe, [30, 80, 140])
tdp["high"] = fuzz.trimf(tdp.universe, [120, 250, 250])

fan_speed["low"] = fuzz.trimf(fan_speed.universe, [0, 0, 30])
fan_speed["medium"] = fuzz.trimf(fan_speed.universe, [20, 50, 80])
fan_speed["high"] = fuzz.trimf(fan_speed.universe, [70, 100, 100])

# Visualize these universes and membership functions
# temp.view()
# load.view()
# tdp.view()
# fan_speed.view()

## Rule Base
#
#   | Temp  | TDP   | Load  | FAN_SPEED |
#   | ----- | ----- | ----- | --------- |
#   | Hi    | -     | -     | Hi        |
#   | Lo    | -     | -     | Lo        |
#   | Md    | Lo    | Lo    | Lo        |
#   | Md    | Hi    | Hi    | Hi        |
#   | Md    | Md    | Md    | Md        |
#   | Lo    | Hi    | Hi    | Md        |
#

rule1 = ctrl.Rule(temp["high"], fan_speed["high"])
rule2 = ctrl.Rule(temp["low"], fan_speed["low"])
rule3 = ctrl.Rule(temp["medium"] & tdp["low"] & load["high"], fan_speed["low"])
rule4 = ctrl.Rule(temp["medium"] & tdp["high"] & load["high"], fan_speed["high"])
rule5 = ctrl.Rule(temp["medium"] & tdp["medium"] & load["high"], fan_speed["medium"])
rule6 = ctrl.Rule(temp["low"] & tdp["high"] & load["high"], fan_speed["medium"])

# rule2.view()
# input()
fan_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])  # , rule4])#, rule5, rule6])
fan = ctrl.ControlSystemSimulation(fan_ctrl)

pygame.init()
pygame.font.init()  # you have to call this at the start,
# if you want to use this module.
my_font = pygame.font.SysFont("Comic Sans MS", 18)
win = pygame.display.set_mode((1000, 600))

text_surface_temp = my_font.render("temp", False, (0, 0, 0))
text_surface_load = my_font.render("load", False, (0, 0, 0))
text_surface_tdp = my_font.render("tdp", False, (0, 0, 0))

slider_temp = Slider(win, 100, 10, 800, 10, min=30, max=89, step=1)
slider_load = Slider(win, 100, 30, 800, 10, min=0, max=100, step=1)
slider_tdp = Slider(win, 100, 50, 800, 10, min=0, max=250, step=1)

output_temp = TextBox(win, 910, 0, 80, 27, fontSize=16)
output_temp.disable()  # Act as label instead of textbox
output_load = TextBox(win, 910, 28, 80, 27, fontSize=16)
output_load.disable()  # Act as label instead of textbox
output_tdp = TextBox(win, 910, 50, 80, 27, fontSize=16)
output_tdp.disable()  # Act as label instead of textbox

run = True
while run:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            run = False
            quit()

    win.fill((255, 255, 255))

    output_load.setText(str(slider_load.getValue()))
    output_temp.setText(str(slider_temp.getValue()))
    output_tdp.setText(str(slider_tdp.getValue()))

    win.blit(text_surface_load, (0, 0))
    win.blit(text_surface_temp, (0, 20))
    win.blit(text_surface_tdp, (0, 40))

    fan.input["temp"] = slider_temp.getValue()
    fan.input["tdp"] = slider_tdp.getValue()
    fan.input["load"] = slider_load.getValue()

    fan.compute()
    try:
        print(fan.output["fan_speed"])
    except:
        print("err")

    pygame_widgets.update(events)
    pygame.display.update()


fig, ax = FuzzyVariableVisualizer(fan_speed).view(sim=fan)
