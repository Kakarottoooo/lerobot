#!/usr/bin/env python3
"""
Test script for Dual-Arm SO-101 MuJoCo Model

This script verifies that your dual-arm model loads correctly.

Usage:
    cd C:\Users\Gzw19\lerobot
    python dual/scripts/test_model.py

Or with a specific model path:
    python dual/scripts/test_model.py --model path/to/your/model.xml
"""

import argparse
import os
import sys

def test_model(xml_path: str):
    """Test if the MuJoCo model loads correctly."""
    
    print(f"\n{'='*60}")
    print("Testing Dual-Arm SO-101 Model")
    print(f"{'='*60}\n")
    
    # Check if file exists
    if not os.path.exists(xml_path):
        print(f"❌ Error: Model file not found: {xml_path}")
        print("\nMake sure you have:")
        print("  1. Cloned lerobot-kinematics to get the SO-101 model")
        print("  2. Created dual_so101_scene.xml in dual/assets/")
        return False
    
    print(f"📁 Loading model: {xml_path}")
    
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        print("✅ Model loaded successfully!\n")
    except ImportError:
        print("❌ MuJoCo not installed. Run: pip install mujoco")
        return False
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Print model info
    print(f"{'Model Information':=^60}")
    print(f"  Number of bodies: {model.nbody}")
    print(f"  Number of joints: {model.njnt}")
    print(f"  Number of actuators: {model.nu}")
    print(f"  Number of sensors: {model.nsensor}")
    print(f"  Degrees of freedom (nv): {model.nv}")
    print(f"  Position state dim (nq): {model.nq}")
    
    # List joints
    print(f"\n{'Joints':=^60}")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            print(f"  [{i:2d}] {name}")
    
    # List actuators
    print(f"\n{'Actuators':=^60}")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            ctrl_range = model.actuator_ctrlrange[i]
            print(f"  [{i:2d}] {name:30s} range: [{ctrl_range[0]:.2f}, {ctrl_range[1]:.2f}]")
    
    # List cameras
    print(f"\n{'Cameras':=^60}")
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if name:
            print(f"  [{i}] {name}")
    
    # Test simulation step
    print(f"\n{'Simulation Test':=^60}")
    mujoco.mj_step(model, data)
    print("✅ Simulation step successful")
    
    print(f"\n{'='*60}")
    print("✅ All tests passed!")
    print(f"{'='*60}\n")
    
    return True


def visualize_model(xml_path: str):
    """Launch MuJoCo viewer to visualize the model."""
    print("\n🎮 Launching MuJoCo viewer...")
    print("   Controls:")
    print("   - Left mouse: Rotate view")
    print("   - Right mouse: Pan view")
    print("   - Scroll: Zoom")
    print("   - ESC: Exit\n")
    
    try:
        import mujoco
        import mujoco.viewer
        
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        
        mujoco.viewer.launch(model, data)
    except Exception as e:
        print(f"❌ Failed to launch viewer: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test Dual-Arm SO-101 Model")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to MJCF model file"
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Launch MuJoCo viewer after testing"
    )
    
    args = parser.parse_args()
    
    # Default model path
    if args.model is None:
        # Try to find the model in expected locations
        possible_paths = [
            "dual/assets/dual_so101_scene.xml",
            "dual/assets/so101_scene.xml",
            "../dual/assets/dual_so101_scene.xml",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                args.model = path
                break
        
        if args.model is None:
            print("❌ No model file found. Please specify with --model")
            print("\nExpected locations:")
            for p in possible_paths:
                print(f"  - {p}")
            print("\nOr create dual_so101_scene.xml following the guide.")
            sys.exit(1)
    
    success = test_model(args.model)
    
    if success and args.visualize:
        visualize_model(args.model)
    elif success:
        response = input("\n🎮 Visualize the model? (y/n): ").strip().lower()
        if response == 'y':
            visualize_model(args.model)


if __name__ == "__main__":
    main()