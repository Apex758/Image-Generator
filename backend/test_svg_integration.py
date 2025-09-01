#!/usr/bin/env python3
"""
Test script to verify SVG generation and processing integration.
"""

import os
import sys
import json
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def test_svg_service_initialization():
    """Test that SVGService initializes correctly."""
    print("Testing SVGService initialization...")
    
    try:
        from main import SVGService
        svg_service = SVGService()
        
        # Check that template metadata is loaded
        assert len(svg_service.template_metadata) == 4
        assert "image_comprehension" in svg_service.template_metadata
        assert "comic" in svg_service.template_metadata
        assert "math" in svg_service.template_metadata
        assert "worksheet" in svg_service.template_metadata
        
        print("✓ SVGService initialization test passed")
        return True
        
    except Exception as e:
        print(f"✗ SVGService initialization test failed: {e}")
        return False

def test_template_loading():
    """Test that SVG templates can be loaded."""
    print("Testing SVG template loading...")
    
    try:
        from main import SVGService
        svg_service = SVGService()
        
        # Test loading each template
        for template_id in svg_service.template_metadata.keys():
            template_content = svg_service.load_template(template_id)
            assert template_content.startswith('<?xml version="1.0"')
            assert '<svg' in template_content
            assert 'placeholder_' in template_content
            
        print("✓ Template loading test passed")
        return True
        
    except Exception as e:
        print(f"✗ Template loading test failed: {e}")
        return False

def test_get_available_templates():
    """Test that available templates can be retrieved."""
    print("Testing get_available_templates...")
    
    try:
        from main import SVGService
        svg_service = SVGService()
        
        templates = svg_service.get_available_templates()
        assert len(templates) == 4
        
        # Check that each template has required fields
        for template in templates:
            assert hasattr(template, 'id')
            assert hasattr(template, 'name')
            assert hasattr(template, 'description')
            assert hasattr(template, 'content_type')
            assert hasattr(template, 'placeholder_count')
            
        print("✓ get_available_templates test passed")
        return True
        
    except Exception as e:
        print(f"✗ get_available_templates test failed: {e}")
        return False

def test_placeholder_extraction():
    """Test that placeholders can be extracted from SVG content."""
    print("Testing placeholder extraction...")
    
    try:
        from main import SVGService
        svg_service = SVGService()
        
        # Load a template and extract placeholders
        svg_content = svg_service.load_template("image_comprehension")
        placeholders = svg_service.extract_placeholders(svg_content)
        
        # Should find some placeholders
        assert len(placeholders) > 0
        
        # Check that common placeholders are found
        expected_placeholders = ["title", "subject", "image"]
        for placeholder in expected_placeholders:
            assert placeholder in placeholders, f"Expected placeholder '{placeholder}' not found"
            
        print(f"✓ Placeholder extraction test passed - found {len(placeholders)} placeholders")
        return True
        
    except Exception as e:
        print(f"✗ Placeholder extraction test failed: {e}")
        return False

def test_text_replacement():
    """Test that text placeholders can be replaced."""
    print("Testing text placeholder replacement...")
    
    try:
        from main import SVGService
        svg_service = SVGService()
        
        # Load a template
        svg_content = svg_service.load_template("image_comprehension")
        
        # Define test replacements
        replacements = {
            "title": "Test Worksheet",
            "subject": "Science - Grade 5th Grade",
            "instructions": "Complete all questions carefully"
        }
        
        # Replace placeholders
        processed_svg = svg_service.replace_text_placeholders(svg_content, replacements)
        
        # Check that replacements were made
        assert "Test Worksheet" in processed_svg
        assert "Science - Grade 5th Grade" in processed_svg
        assert "Complete all questions carefully" in processed_svg
        
        print("✓ Text replacement test passed")
        return True
        
    except Exception as e:
        print(f"✗ Text replacement test failed: {e}")
        return False

def test_data_models():
    """Test that data models can be instantiated."""
    print("Testing data models...")
    
    try:
        from main import (
            SVGGenerationRequest, SVGProcessingRequest, SVGExportRequest,
            SVGTemplate, SVGGenerationResponse, SVGProcessingResponse, SVGExportResponse
        )
        
        # Test SVGGenerationRequest
        gen_request = SVGGenerationRequest(
            content_type="math",
            subject="Mathematics",
            grade="3rd Grade",
            prompt="Create a math worksheet about addition"
        )
        assert gen_request.content_type == "math"
        assert gen_request.image_count == 1  # default value
        
        # Test SVGTemplate
        template = SVGTemplate(
            id="test",
            name="Test Template",
            description="A test template",
            content_type="test",
            placeholder_count=5
        )
        assert template.id == "test"
        
        print("✓ Data models test passed")
        return True
        
    except Exception as e:
        print(f"✗ Data models test failed: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist."""
    print("Testing directory structure...")
    
    try:
        from main import SVG_TEMPLATES_DIR, SVG_EXPORTS_DIR, IMAGES_DIR
        
        # Check that directories exist
        assert os.path.exists(SVG_TEMPLATES_DIR), f"SVG templates directory not found: {SVG_TEMPLATES_DIR}"
        assert os.path.exists(IMAGES_DIR), f"Images directory not found: {IMAGES_DIR}"
        
        # SVG_EXPORTS_DIR might not exist yet, but the parent should
        parent_dir = os.path.dirname(SVG_EXPORTS_DIR)
        assert os.path.exists(parent_dir), f"Parent directory for exports not found: {parent_dir}"
        
        # Check that template files exist
        template_files = ["image_comprehension.svg", "comic.svg", "math.svg", "worksheet.svg"]
        for template_file in template_files:
            template_path = os.path.join(SVG_TEMPLATES_DIR, template_file)
            assert os.path.exists(template_path), f"Template file not found: {template_path}"
            
        print("✓ Directory structure test passed")
        return True
        
    except Exception as e:
        print(f"✗ Directory structure test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== SVG Integration Test Suite ===\n")
    
    tests = [
        test_directory_structure,
        test_svg_service_initialization,
        test_data_models,
        test_template_loading,
        test_get_available_templates,
        test_placeholder_extraction,
        test_text_replacement,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! SVG integration is working correctly.")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())