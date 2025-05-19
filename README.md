This repository contains code for detecting, classifying, and locating faults in the IEEE 9-bus power system using machine learning.
Objectives:
- Detects whether a fault is present in the power grid
- Identifies the specific transmission line where the fault occurred
- Classifies the type of fault (e.g., LG, LLG, LLL)
- Predicts the exact location of the fault along the line

Input features:
- 3-phase Voltage and Current measurements
- Data collected from both ends of power lines
- Simulated fault scenarios injected at various locations along the line

ML approach used:
- Used Random Forest classifiers
 -Separate models trained for:
  - Fault detection
  - Faut identification
  - Fault classification
  - Fault location prediction

"fault_detection_script.py": Main Python script containing the ML pipeline and fault analysis logic
