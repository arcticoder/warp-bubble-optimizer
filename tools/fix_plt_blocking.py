#!/usr/bin/env python3
"""
Script to replace plt.close()  # Close instead of show to prevent blocking with plt.close() to prevent blocking visualization windows
"""

import os
import re
from pathlib import Path

def fix_plt_show_in_file(filepath):
    """Fix plt.close()  # Close instead of show to prevent blocking calls in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count original plt.close()  # Close instead of show to prevent blocking calls
        original_count = len(re.findall(r'plt\.show\(\)', content))
        
        if original_count == 0:
            return 0, 0  # No changes needed
        
        # Replace plt.close()  # Close instead of show to prevent blocking with plt.close() and add comment
        new_content = re.sub(
            r'(\s*)plt\.show\(\)',
            r'\1plt.close()  # Close instead of show to prevent blocking',
            content
        )
        
        # Write back the modified content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        new_count = len(re.findall(r'plt\.show\(\)', new_content))
        fixed_count = original_count - new_count
        
        return original_count, fixed_count
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return 0, 0

def main():
    """Main function to fix plt.close()  # Close instead of show to prevent blocking across all Python files"""
    print("🔧 Fixing plt.close()  # Close instead of show to prevent blocking blocking issues across all Python files...")
    print("=" * 60)
    
    # Find all Python files
    python_files = list(Path('.').glob('**/*.py'))
    
    total_original = 0
    total_fixed = 0
    files_modified = 0
    
    for filepath in python_files:
        original, fixed = fix_plt_show_in_file(filepath)
        if fixed > 0:
            files_modified += 1
            print(f"✅ {filepath}: Fixed {fixed}/{original} plt.close()  # Close instead of show to prevent blocking calls")
        
        total_original += original
        total_fixed += fixed
    
    print("\n" + "=" * 60)
    print(f"📊 SUMMARY:")
    print(f"   Files scanned: {len(python_files)}")
    print(f"   Files modified: {files_modified}")
    print(f"   Total plt.close()  # Close instead of show to prevent blocking calls found: {total_original}")
    print(f"   Total plt.close()  # Close instead of show to prevent blocking calls fixed: {total_fixed}")
    
    if total_fixed > 0:
        print(f"\n✅ SUCCESS: All plt.close()  # Close instead of show to prevent blocking calls have been replaced with plt.close()")
        print(f"   Scripts will no longer block on visualization windows!")
    else:
        print(f"\n✅ No plt.close()  # Close instead of show to prevent blocking calls found - all files are already non-blocking!")

if __name__ == "__main__":
    main()
