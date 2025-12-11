import mujoco
import mujoco.viewer
import os

# Path to your XML file
xml_path = "dual/assets/dual_so101_scene.xml"

# Check if file exists
if not os.path.exists(xml_path):
    # Try finding it relative to where we run the script
    xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "dual_so101_scene.xml")

print(f"Loading: {xml_path}")

# Load the model directly
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Launch the native interactive viewer
# This automatically handles the physics loop for you
print("Launching viewer...")
print("1. Press SPACE to pause/unpause physics.")
print("2. Expand 'Control' on the right to move joints.")
mujoco.viewer.launch(model, data)