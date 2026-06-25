#!/usr/bin/env python3
"""
Test script for the Energy Demand Forecasting API.
"""

import requests
import json
import time

def test_api():
    """Test the API endpoints."""
    base_url = "http://localhost:8000"
    
    print("Testing Energy Demand Forecasting API...")
    print("=" * 50)
    
    # Test 1: Check if server is running
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure it's running on localhost:8000")
        return
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        return
    
    # Test 2: Get initial data
    try:
        response = requests.get(f"{base_url}/get_initial_data", timeout=5)
        data = response.json()
        if "initial_data" in data:
            print(f"✅ Initial data endpoint working - got {len(data['initial_data'])} data points")
        else:
            print(f"❌ Initial data endpoint error: {data.get('error', 'Unknown error')}")
            return
    except Exception as e:
        print(f"❌ Error getting initial data: {e}")
        return
    
    # Test 3: Test prediction endpoint
    try:
        # Use the initial data for prediction
        test_sequence = data['initial_data']
        response = requests.post(
            f"{base_url}/predict",
            json={"sequence": test_sequence},
            timeout=10
        )
        pred_data = response.json()
        if "prediction" in pred_data:
            print(f"✅ Prediction endpoint working - prediction: {pred_data['prediction']:.4f} kW")
        else:
            print(f"❌ Prediction endpoint error: {pred_data.get('error', 'Unknown error')}")
            return
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return
    
    # Test 4: Test get next actual
    try:
        response = requests.get(f"{base_url}/get_next_actual", timeout=5)
        actual_data = response.json()
        if "next_actual" in actual_data:
            print(f"✅ Next actual endpoint working - next value: {actual_data['next_actual']:.4f} kW")
        else:
            print(f"❌ Next actual endpoint error: {actual_data.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Error getting next actual: {e}")
    
    print("=" * 50)
    print("✅ All API tests completed successfully!")
    print("\nTo start the server, run:")
    print("  python src/api/main.py")
    print("  or")
    print("  uvicorn src.api.main:app --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    test_api()