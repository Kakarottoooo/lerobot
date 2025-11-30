Project: Resolved-Rate Control of Stacked LeRobot Arms
Student: Ziwei Guo

Requirements:
- Python 3.10+
- numpy < 2.0
- roboticstoolbox-python
- matplotlib

Folder Structure:
- project/
  - solver.py        : Core math class (Jacobians + Weighted Inverse)
  - visualize.py     : 3D Animation script (Matplotlib)
  - plots.py         : Generates the comparison graphs
  - so101.urdf       : Robot definition file
  - assets/          : 3D meshes for the robot

How to Run:
1. Comparison Graphs (The Main Result):
   Run: python plots.py
   This will generate 'comparison_results.png' showing the difference 
   between standard and weighted control.

2. 3D Simulation:
   Run: python visualize.py
   This will open a window showing the stacked robots moving in 3D space.

3. Core Logic:
   The math implementation is found in 'solver.py' inside the 
   'StackedResolvedRateControl' class.