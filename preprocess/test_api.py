#!/usr/bin/env python3
"""
Test script for Google Gemini API integration.
"""

import os
import json
from google import genai
from google.genai import types

def test_api():
    """Test the Google Gemini API with proxy."""
    # Configuration
    api_key = "sk-LcoHn73XDQIN6RqS1Fwp90Ec6ZJ3IO85CBDqDMUOdpuallDq"
    base_url = "https://api.openai-proxy.org/google"
    model = "gemini-2.0-flash"
    
    print(f"Testing API with:")
    print(f"  - Base URL: {base_url}")
    print(f"  - Model: {model}")
    print(f"  - API Key: {api_key[:10]}...")
    
    try:
        # Set up client with proxy
        os.environ['API_KEY'] = api_key
        http_options = types.HttpOptions(base_url=base_url)
        client = genai.Client(api_key=api_key, http_options=http_options)
        
        # Simple text prompt
        test_prompt = "Hello! Please respond with 'API test successful' if you can see this message."
        
        print("\n🚀 Sending test request...")
        
        # Generate content
        response = client.models.generate_content(
            model=model,
            contents=[test_prompt]
        )
        
        if response and response.text:
            print(f"✅ API Response: {response.text}")
            return True
        else:
            print("❌ Empty response from API")
            return False
            
    except Exception as e:
        print(f"❌ API Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_api()
    exit(0 if success else 1)
