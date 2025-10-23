import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

## notatki
# na jutro: https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem_newapi.html
# + plus poprawić początek na ten z aktualnego tutoriala

## Fuzzification

# Generate universe variables
x_temp = np.arange(30, 90, 0.1) # in C
x_load = np.arange(0, 101, 1) # in %
x_tdp = np.arange(0, 250, 1) # in Watts

x_fan_speed = np.arange(0, 101, 1) # (out) %

# Generate fuzzy membership functions

temp_lo = fuzz.trimf(x_temp, [30, 30, 50])
temp_md = fuzz.trimf(x_temp, [40, 60, 80])
temp_hi = fuzz.trimf(x_temp, [70, 90, 90])
tdp_lo = fuzz.trimf(x_tdp, [0, 0, 40])
tdp_md = fuzz.trimf(x_tdp, [30, 80, 140])
tdp_hi = fuzz.trimf(x_tdp, [120, 250, 250])
load_lo = fuzz.trimf(x_load, [0, 0, 30])
load_md = fuzz.trimf(x_load, [20, 50, 80])
load_hi = fuzz.trimf(x_load, [70, 100, 100])

fan_speed_lo = fuzz.trimf(x_load, [0, 0, 30])
fan_speed_md = fuzz.trimf(x_load, [20, 50, 80])
fan_speed_hi = fuzz.trimf(x_load, [70, 100, 100])

# Visualize these universes and membership functions
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_temp, temp_lo, "b", linewidth=1.5, label="Low")
ax0.plot(x_temp, temp_md, "g", linewidth=1.5, label="Medium")
ax0.plot(x_temp, temp_hi, "r", linewidth=1.5, label="High")
ax0.set_title("CPU Temperature")
ax0.legend()

ax1.plot(x_load, load_lo, "b", linewidth=1.5, label="Low")
ax1.plot(x_load, load_md, "g", linewidth=1.5, label="Medium")
ax1.plot(x_load, load_hi, "r", linewidth=1.5, label="High")
ax1.set_title("CPU Load")
ax1.legend()

ax2.plot(x_tdp, tdp_lo, "b", linewidth=1.5, label="Low")
ax2.plot(x_tdp, tdp_md, "g", linewidth=1.5, label="Medium")
ax2.plot(x_tdp, tdp_hi, "r", linewidth=1.5, label="High")
ax2.set_title("CPU TDP")
ax2.legend()

# plt.tight_layout()
# plt.show()

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

rule1 = ctrl.Rule(temp_hi, fan_speed_hi)
rule2 = ctrl.Rule(temp_lo, fan_speed_lo)
rule3 = ctrl.Rule(temp_md & tdp_lo & load_hi, fan_speed_lo)
rule4 = ctrl.Rule(temp_md & tdp_hi & load_hi, fan_speed_hi)
rule5 = ctrl.Rule(temp_md & tdp_md & load_hi, fan_speed_md)
rule6 = ctrl.Rule(temp_lo & tdp_hi & load_hi, fan_speed_md)

rule1.view()

fan_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
fan = ctrl.ControlSystemSimulation(fan_ctrl)
