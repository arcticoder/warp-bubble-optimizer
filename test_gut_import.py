import os
import sys

# Add the path to unified_gut_polymerization package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "unified-gut-polymerization")))

print("Python path:")
print("\n".join(sys.path))

print("\nAttempting to import unified_gut_polymerization:")
try:
    import unified_gut_polymerization
    print('Import succeeded!')
    
    # List available attributes
    print("\nPackage attributes:")
    attrs = dir(unified_gut_polymerization)
    for attr in attrs:
        if not attr.startswith('__'):
            print(f"- {attr}")
    
    # Check core module
    print("\nChecking core module:")
    try:
        from unified_gut_polymerization.core import UnifiedGaugePolymerization, GUTConfig
        print("Core module classes imported successfully")
        
        # Create an instance to test
        config = GUTConfig(group='SU5', polymer_length=0.1)
        polymer = UnifiedGaugePolymerization(config)
        print(f"Successfully created UnifiedGaugePolymerization instance for {config.group}")
        print(f"Polymer length: {polymer.config.polymer_length}")
    except (ImportError, AttributeError) as e:
        print(f"Core module error: {e}")
except ImportError as e:
    print(f'Import failed: {e}')
