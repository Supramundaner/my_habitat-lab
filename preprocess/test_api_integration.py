#!/usr/bin/env python3
"""
Test script for Gemini API integration using REST API.
Tests the API connection and basic functionality.
"""

import os
import json
import requests
import base64
import tempfile
import cv2
import numpy as np

def test_gemini_api():
    """Test basic Gemini API functionality."""
    
    # Configuration
    config = {
        "llm_config": {
            "api_key": "AIzaSyB_l2MZK7aGKP8AxixoTbWepcCSGYJnZ_4",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/models",
            "model": "gemini-2.0-flash",
            "max_tokens": 1000
        }
    }
    
    print("🧪 Testing Gemini API Integration")
    print("="*50)
    
    # Test 1: Text-only request
    print("\n1. Testing text-only request...")
    try:
        llm_config = config['llm_config']
        api_key = llm_config['api_key']
        base_url = llm_config['base_url']
        model = llm_config['model']
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": "Explain how AI works in a few words"
                        }
                    ]
                }
            ]
        }
        
        headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': api_key
        }
        
        api_url = f"{base_url}/{model}:generateContent"
        print(f"   API URL: {api_url}")
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            response_data = response.json()
            if 'candidates' in response_data and response_data['candidates']:
                text_response = response_data['candidates'][0]['content']['parts'][0]['text']
                print(f"   ✅ Success! Response: {text_response[:100]}...")
            else:
                print(f"   ❌ No candidates in response: {response_data}")
                return False
        else:
            print(f"   ❌ API request failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Text test failed: {e}")
        return False
    
    # Test 2: Image + text request
    print("\n2. Testing image + text request...")
    try:
        # Create a simple test image
        test_image = np.ones((200, 300, 3), dtype=np.uint8) * 255  # White image
        cv2.rectangle(test_image, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue square
        cv2.putText(test_image, "TEST", (80, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save to temporary file and encode
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            cv2.imwrite(temp_file.name, test_image)
            temp_path = temp_file.name
        
        try:
            # Encode image to base64
            with open(temp_path, 'rb') as f:
                image_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": "What do you see in this image? Respond in one word."
                            },
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": image_b64
                                }
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                response_data = response.json()
                if 'candidates' in response_data and response_data['candidates']:
                    text_response = response_data['candidates'][0]['content']['parts'][0]['text']
                    print(f"   ✅ Success! Image response: {text_response[:100]}...")
                else:
                    print(f"   ❌ No candidates in image response: {response_data}")
                    return False
            else:
                print(f"   ❌ Image API request failed: {response.status_code} - {response.text}")
                return False
                
        finally:
            # Clean up temp file
            os.unlink(temp_path)
            
    except Exception as e:
        print(f"   ❌ Image test failed: {e}")
        return False
    
    print("\n✅ All API tests passed!")
    return True

def test_workflow_config():
    """Test that the workflow configuration is valid."""
    print("\n🔧 Testing workflow configuration...")
    
    try:
        # Load the actual config file
        config_path = "input_config.json"
        if not os.path.exists(config_path):
            print(f"   ❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check required fields
        required_fields = [
            ['scene_config', 'scene_path'],
            ['llm_config', 'api_key'],
            ['llm_config', 'base_url'],
            ['llm_config', 'model'],
            ['prompts', 'choose_room_prompt'],
            ['prompts', 'choose_node_prompt'],
            ['output', 'output_dir']
        ]
        
        for field_path in required_fields:
            current = config
            for field in field_path:
                if field not in current:
                    print(f"   ❌ Missing field: {'.'.join(field_path)}")
                    return False
                current = current[field]
        
        # Check if scene file exists
        scene_path = config['scene_config']['scene_path']
        if not os.path.exists(scene_path):
            print(f"   ⚠️ Scene file not found: {scene_path}")
            print(f"      (This is expected if you haven't set up the scene yet)")
        
        # Check if prompt files exist
        for prompt_key in ['choose_room_prompt', 'choose_node_prompt']:
            prompt_path = config['prompts'][prompt_key]
            if not os.path.exists(prompt_path):
                print(f"   ❌ Prompt file not found: {prompt_path}")
                return False
        
        print(f"   ✅ Configuration is valid!")
        print(f"   📝 Model: {config['llm_config']['model']}")
        print(f"   🔗 Base URL: {config['llm_config']['base_url']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Config test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Gemini API Integration Test Suite")
    print("="*60)
    
    success = True
    
    # Test API functionality
    if not test_gemini_api():
        success = False
    
    # Test workflow configuration
    if not test_workflow_config():
        success = False
    
    print("\n" + "="*60)
    if success:
        print("🎉 All tests passed! The API integration is ready.")
        print("\nYou can now run the workflow:")
        print("python main_workflow.py input_config.json")
    else:
        print("❌ Some tests failed. Please check the configuration and API key.")
    
    print("="*60)
