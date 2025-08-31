import requests
import json
import sys

# Sample lesson plan from mockData.ts
sample_lesson_plan = """
Objective: Students will understand the concept of fractions as parts of a whole.

Materials: Fraction circles, worksheets, interactive whiteboard.

Activities:
1. Begin by showing students a whole pizza and then cutting it into equal parts.
2. Introduce the terms numerator and denominator.
3. Have students practice identifying fractions using visual models.
4. Group activity: Students create their own fraction models using paper plates.

Assessment: Students will complete a worksheet identifying fractions in various models.
"""

def test_analyze_lesson_plan():
    """Test the lesson plan analysis endpoint"""
    print("Testing /api/analyze-lesson-plan endpoint...")
    
    url = "http://localhost:8000/api/analyze-lesson-plan"
    headers = {"Content-Type": "application/json"}
    payload = {
        "lesson_plan": sample_lesson_plan,
        "max_images": 5
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\nGenerated Image Prompts:")
            for i, prompt in enumerate(result["image_prompts"], 1):
                print(f"\n--- Prompt {i} ---")
                print(f"Prompt: {prompt['prompt']}")
                print(f"Explanation: {prompt['explanation']}")
            
            print(f"\nTotal prompts generated: {len(result['image_prompts'])}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return False

if __name__ == "__main__":
    # Start the server separately with: uvicorn main:app --reload
    success = test_analyze_lesson_plan()
    sys.exit(0 if success else 1)