"""
Test script to verify Google Gemini API is working.

Usage:
    # Set your API key first
    export GEMINI_API_KEY=your_key_here

    # Run the test
    uv run LLM/test_gemini.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    print("❌ google-genai package not installed")
    print("Run: uv pip install google-genai")
    sys.exit(1)

# Load environment
load_dotenv()

def test_gemini_connection():
    """Test basic Gemini API connection and generation."""

    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        print("\nAdd to your .env file:")
        print("GEMINI_API_KEY=your_actual_key_here")
        return False

    print("✓ API key found")

    # Initialize client
    try:
        client = genai.Client(api_key=api_key)
        print("✓ Client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return False

    # Test model
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    print(f"✓ Using model: {model_name}")

    # Test generation
    try:
        print("\n🔄 Testing text generation...")
        response = client.models.generate_content(
            model=model_name,
            contents="Say 'Hello from Gemini!' and nothing else.",
            config={
                "max_output_tokens": 50,
                "temperature": 0.1,
            }
        )

        result_text = response.text.strip()
        print(f"✓ Response received: {result_text}")

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

    # Test JSON generation (for stance extraction)
    try:
        print("\n🔄 Testing JSON generation...")
        response = client.models.generate_content(
            model=model_name,
            contents="""Return a JSON object with this structure:
{
  "test": "success",
  "message": "Gemini is working"
}

Respond ONLY with the JSON, no markdown or extra text.""",
            config={
                "max_output_tokens": 100,
                "temperature": 0.1,
            }
        )

        import json
        result = response.text.strip()

        # Handle potential markdown wrapping
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()

        parsed = json.loads(result)
        print(f"✓ JSON response: {parsed}")

        if parsed.get("test") == "success":
            print("✓ JSON parsing successful")
        else:
            print("⚠️  JSON parsed but unexpected content")

    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        print(f"Raw response: {response.text}")
        return False
    except Exception as e:
        print(f"❌ JSON generation failed: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Google Gemini API Test")
    print("=" * 60)
    print()

    success = test_gemini_connection()

    print()
    print("=" * 60)
    if success:
        print("✅ All tests passed! Gemini is ready to use.")
        print()
        print("To use Gemini for stance extraction, add to .env:")
        print("  LLM_PROVIDER=gemini")
    else:
        print("❌ Tests failed. Check your API key and configuration.")
    print("=" * 60)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()