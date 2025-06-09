#!/usr/bin/env python3
"""
Simple Integration Test
======================

Basic test to verify the integrated control system works.
"""

def test_simple():
    print("Testing simple import...")
    try:
        from integrated_impulse_control import IntegratedImpulseController, MissionWaypoint, ImpulseEngineConfig
        from simulate_vector_impulse import Vector3D
        from simulate_rotation import Quaternion
        print("✅ Imports successful")
        
        # Test basic instantiation
        config = ImpulseEngineConfig()
        controller = IntegratedImpulseController(config)
        print("✅ Controller created")
        
        # Test waypoint creation
        waypoint = MissionWaypoint(
            position=Vector3D(100.0, 0.0, 0.0),
            orientation=Quaternion(1.0, 0.0, 0.0, 0.0)
        )
        print("✅ Waypoint created")
        
        print("🎉 Basic integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple()
    exit(0 if success else 1)
