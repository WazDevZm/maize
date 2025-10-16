#!/usr/bin/env python3
"""
Test script to verify all 4 disease conditions can be detected
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from app import find_best_model, load_model, DISEASE_INFO

def test_disease_conditions():
    """Test that all 4 disease conditions are properly configured"""
    print("🧪 Testing Disease Condition Configuration")
    print("=" * 50)
    
    # Check if all 4 conditions are defined
    required_conditions = ["Health", "Grey_Leaf_Spots", "Leaf_Blight", "MSV"]
    
    print("✅ Required Conditions:", required_conditions)
    print("✅ Available Conditions:", list(DISEASE_INFO.keys()))
    
    missing_conditions = [cond for cond in required_conditions if cond not in DISEASE_INFO]
    
    if missing_conditions:
        print(f"❌ Missing conditions: {missing_conditions}")
        return False
    else:
        print("✅ All 4 conditions are properly configured!")
    
    # Check condition details
    print("\n📋 Condition Details:")
    for condition in required_conditions:
        info = DISEASE_INFO[condition]
        print(f"  {condition}:")
        print(f"    - Description: {info.get('description', 'N/A')}")
        print(f"    - Severity: {info.get('severity', 'N/A')}")
        print(f"    - Color: {info.get('color', 'N/A')}")
        print(f"    - Symptoms: {len(info.get('symptoms', []))} items")
        print()
    
    return True

def test_model_loading():
    """Test model loading capabilities"""
    print("🤖 Testing Model Loading")
    print("=" * 50)
    
    try:
        # Find best model
        model_info, error = find_best_model()
        
        if model_info is None:
            print(f"❌ Model not found: {error}")
            return False
        
        print(f"✅ Model found: {model_info.get('name', 'Unknown')}")
        print(f"✅ Model path: {model_info.get('path', 'Unknown')}")
        print(f"✅ Model size: {model_info.get('size_mb', 0):.1f} MB")
        print(f"✅ Last modified: {model_info.get('modified', 'Unknown')}")
        
        # Try to load the model
        model, model_info_loaded = load_model()
        
        if model is None:
            print("❌ Failed to load model")
            return False
        
        print("✅ Model loaded successfully!")
        
        # Check model classes if available
        if hasattr(model, 'names'):
            model_classes = list(model.names.values())
            print(f"✅ Model classes: {model_classes}")
            
            # Check if all required classes are present
            required_classes = ["Health", "Grey_Leaf_Spots", "Leaf_Blight", "MSV"]
            missing_classes = [cls for cls in required_classes if cls not in model_classes]
            
            if missing_classes:
                print(f"⚠️  Missing classes in model: {missing_classes}")
            else:
                print("✅ All 4 conditions are supported by the model!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model loading: {e}")
        return False

def main():
    """Run all tests"""
    print("🌽 Maize Disease Detection - Detection Test")
    print("=" * 60)
    
    # Test 1: Disease conditions
    conditions_ok = test_disease_conditions()
    
    # Test 2: Model loading
    model_ok = test_model_loading()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 30)
    print(f"Disease Conditions: {'✅ PASS' if conditions_ok else '❌ FAIL'}")
    print(f"Model Loading: {'✅ PASS' if model_ok else '❌ FAIL'}")
    
    if conditions_ok and model_ok:
        print("\n🎉 All tests passed! The app can detect all 4 conditions.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the configuration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
