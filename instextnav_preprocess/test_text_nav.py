#!/usr/bin/env python3
"""
Test script for TextNav modifications.
Tests the text description loading and basic workflow.
"""

import os
import sys
from text_description_loader import TextDescriptionLoader

def test_text_description_loader():
    """Test the text description loader functionality."""
    print("🧪 Testing Text Description Loader...")
    
    # Test with the example data
    val_text_path = "val_text.json"
    scene_path = "/home/yaoaa/habitat-lab/data/versioned_data/hm3d-0.2/hm3d/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb"
    object_id = "8"
    object_category = "chair"
    
    try:
        # Test loader initialization
        loader = TextDescriptionLoader(val_text_path)
        print("✓ TextDescriptionLoader initialized successfully")
        
        # Test scene ID extraction
        scene_id = loader.extract_scene_id_from_path(scene_path)
        print(f"✓ Scene ID extracted: {scene_id}")
        
        # Test object validation
        exists = loader.validate_object_exists(scene_id, object_id)
        print(f"✓ Object {scene_id}_{object_id} exists: {exists}")
        
        # Test description retrieval
        description_tuple = loader.get_object_description(scene_id, object_id)
        if description_tuple:
            intrinsic, extrinsic = description_tuple
            print(f"✓ Description retrieved:")
            print(f"  - Intrinsic: {intrinsic[:100]}...")
            print(f"  - Extrinsic: {extrinsic[:100]}...")
        
        # Test combined description
        combined = loader.create_combined_description(scene_id, object_id, object_category)
        if combined:
            print(f"✓ Combined description created ({len(combined)} chars)")
            print(f"  Preview: {combined[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading with text nav settings."""
    print("\n🧪 Testing Configuration Loading...")
    
    try:
        import json
        
        # Load the config file
        config_path = "input_config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Check text nav settings
        use_text_nav = config.get('scene_config', {}).get('use_text_nav', False)
        val_text_path = config.get('val_text_path', 'val_text.json')
        
        print(f"✓ Config loaded successfully")
        print(f"  - use_text_nav: {use_text_nav}")
        print(f"  - val_text_path: {val_text_path}")
        
        # Check prompt paths
        prompts = config.get('prompts', {})
        text_prompts_exist = (
            'choose_room_prompt_text' in prompts and 
            'choose_node_prompt_text' in prompts
        )
        print(f"  - Text prompts configured: {text_prompts_exist}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_prompt_templates():
    """Test that text prompt templates exist and are readable."""
    print("\n🧪 Testing Prompt Templates...")
    
    try:
        prompt_files = [
            "prompts/choose_room_prompt_text.txt",
            "prompts/choose_node_prompt_text.txt"
        ]
        
        for prompt_file in prompt_files:
            if os.path.exists(prompt_file):
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"✓ {prompt_file} exists ({len(content)} chars)")
            else:
                print(f"✗ {prompt_file} not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Prompt template test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting TextNav Tests...\n")
    
    tests = [
        test_text_description_loader,
        test_config_loading,
        test_prompt_templates
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n📊 Test Results:")
    print(f"  - Passed: {sum(results)}/{len(results)}")
    print(f"  - Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! TextNav modifications are ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
