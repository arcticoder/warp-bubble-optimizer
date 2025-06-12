import sys

print('Python path:')
print('\n'.join(sys.path))

print('\nAttempting to import unified_gut_polymerization:')
try:
    import unified_gut_polymerization
    print('Import succeeded!')
    if hasattr(unified_gut_polymerization, '__version__'):
        print(f'Version: {unified_gut_polymerization.__version__}')
except ImportError as e:
    print(f'Import failed: {e}')
