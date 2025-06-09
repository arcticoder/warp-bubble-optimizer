#!/usr/bin/env python3
"""
Quick test for backreaction analysis timeout issue
"""

import sys
import os
sys.path.append('.')

from src.warp_engine.backreaction import BackreactionAnalyzer, EinsteinSolver
import time

def test_backreaction_timeout():
    """Test backreaction analysis with timeout"""
    print("🧪 Testing backreaction analysis...")
    
    # Create minimal test setup
    einstein_solver = EinsteinSolver()
    analyzer = BackreactionAnalyzer(einstein_solver)
    
    print("⏱️  Starting backreaction analysis (with 10s timeout)...")
    start_time = time.time()
    
    try:
        result = analyzer.analyze_backreaction_coupling(
            bubble_radius=10.0,
            bubble_speed=1000.0,
            timeout=10.0  # 10 second timeout
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Analysis completed in {elapsed:.2f}s")
        print(f"📊 Results: success={result.get('einstein_success', False)}, "
              f"residual={result.get('max_residual', 'N/A')}")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Analysis failed after {elapsed:.2f}s: {e}")

if __name__ == "__main__":
    test_backreaction_timeout()
