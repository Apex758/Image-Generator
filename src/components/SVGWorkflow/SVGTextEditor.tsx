import React, { useState, useEffect, useMemo } from 'react';
import { useMutation } from 'react-query';
import { Button } from '../ui/Button';
import { Save, X, RotateCcw, Eye, EyeOff } from 'lucide-react';
import { svgApi, ProcessSVGRequest } from '../../lib/api';

// Function to safely clean and prepare SVG content for rendering (same as SVGPreview)
const cleanSVGContent = (svgString: string): string => {
  let content = svgString;

  // Remove any XML declarations if present
  content = content.replace(/<\?xml[^>]*\?>/g, '');
  
  // Ensure proper SVG namespace
  if (!content.includes('xmlns="http://www.w3.org/2000/svg"')) {
    content = content.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"');
  }

  // Clean up any namespace prefixes that might cause issues
  content = content.replace(/ns0:/g, '');
  content = content.replace(/xmlns:ns0="[^"]*"/g, '');

  // Ensure viewBox is present and valid
  if (!content.includes('viewBox=')) {
    const widthMatch = content.match(/width="([^"]+)"/);
    const heightMatch = content.match(/height="([^"]+)"/);
    if (widthMatch && heightMatch) {
      const width = parseInt(widthMatch[1]) || 800;
      const height = parseInt(heightMatch[1]) || 600;
      content = content.replace('<svg', `<svg viewBox="0 0 ${width} ${height}"`);
    }
  }

  return content;
};

interface SVGTextEditorProps {
  svgContent: string;
  placeholders: string[];
  onSave: (processedContent: string) => void;
  onCancel: () => void;
}

export const SVGTextEditor: React.FC<SVGTextEditorProps> = ({
  svgContent,
  placeholders,
  onSave,
  onCancel
}) => {
  // Filter to only show AI-generated content placeholders
  const editablePlaceholders = placeholders.filter(placeholder => {
    const aiContentFields = ['instructions', 'question1', 'question2', 'question3', 'question4', 'question5'];
    return aiContentFields.includes(placeholder);
  });

  const [replacements, setReplacements] = useState<Record<string, string>>({});
  const [showPreview, setShowPreview] = useState(true);
  const [previewContent, setPreviewContent] = useState(svgContent);

  // Extract actual AI-generated content from SVG
  const extractContentFromSVG = (placeholder: string): string => {
    try {
      // Create a temporary DOM element to parse the SVG
      const parser = new DOMParser();
      const doc = parser.parseFromString(svgContent, 'image/svg+xml');
      
      // Look for text elements that might contain our content
      const textElements = doc.querySelectorAll('text, tspan');
      
      for (const element of textElements) {
        const textContent = element.textContent?.trim() || '';
        
        // Skip empty content or template fields
        if (!textContent || textContent.length < 10) continue;
        
        // Match content based on placeholder type
        if (placeholder === 'instructions') {
          // Instructions usually contain words like "look", "answer", "examine", etc.
          if (textContent.toLowerCase().includes('look') ||
              textContent.toLowerCase().includes('answer') ||
              textContent.toLowerCase().includes('examine') ||
              textContent.toLowerCase().includes('study') ||
              textContent.toLowerCase().includes('questions')) {
            return textContent;
          }
        } else if (placeholder.startsWith('question')) {
          // Questions typically end with "?" or contain question words
          if (textContent.includes('?') ||
              textContent.toLowerCase().match(/^(what|how|why|where|when|which|describe|explain|identify)/)) {
            // Try to find the right question by position or content
            if (textContent.includes('?')) {
              return textContent;
            }
          }
        }
      }
      
      // Fallback: return empty string to allow user input
      return '';
    } catch (error) {
      console.warn('Failed to extract content from SVG:', error);
      return '';
    }
  };

  // Initialize replacements with actual AI-generated content from SVG (only on first load)
  useEffect(() => {
    // Only initialize if replacements is empty (first load)
    if (Object.keys(replacements).length === 0) {
      const initialReplacements: Record<string, string> = {};
      editablePlaceholders.forEach(placeholder => {
        const extractedContent = extractContentFromSVG(placeholder);
        initialReplacements[placeholder] = extractedContent || '';
      });
      setReplacements(initialReplacements);
    }
  }, [editablePlaceholders]); // Remove svgContent from dependencies to prevent resets

  // Update preview when replacements change
  useEffect(() => {
    let updatedContent = svgContent;
    
    // Replace actual content in the SVG, not placeholder patterns
    Object.entries(replacements).forEach(([placeholder, replacement]) => {
      if (replacement.trim()) {
        // Get the original content that was extracted for this placeholder
        const originalContent = extractContentFromSVG(placeholder);
        if (originalContent) {
          // Replace the original content with the new replacement
          updatedContent = updatedContent.replace(originalContent, replacement);
        }
      }
    });
    
    setPreviewContent(updatedContent);
  }, [svgContent, replacements]);

  const processMutation = useMutation(
    (request: ProcessSVGRequest) => svgApi.process(request),
    {
      onSuccess: (data) => {
        onSave(data.processed_svg);  // Updated to match new interface
      },
      onError: (error: unknown) => {
        console.error('SVG processing failed:', error);
        alert('Failed to process SVG. Please try again.');
      },
    }
  );

  const handleReplacementChange = (placeholder: string, value: string) => {
    setReplacements(prev => ({
      ...prev,
      [placeholder]: value
    }));
  };

  const handleReset = () => {
    const initialReplacements: Record<string, string> = {};
    editablePlaceholders.forEach(placeholder => {
      initialReplacements[placeholder] = placeholder;
    });
    setReplacements(initialReplacements);
  };

  // Memoize the SVG processing to avoid re-running on every render (same as SVGPreview)
  const processedSVG = useMemo(() => {
    let content = cleanSVGContent(previewContent);

    // Highlight placeholders like in the main preview
    content = content.replace(
      /\[([^\]]+)\]/g,
      '<tspan fill="#ef4444" font-weight="bold">[$1]</tspan>'
    );

    return content;
  }, [previewContent]);

  const handleSave = () => {
    const request: ProcessSVGRequest = {
      svg_content: svgContent,
      text_replacements: replacements,  // Changed from 'replacements' to 'text_replacements'
      add_writing_lines: false  // Added the optional field
    };
    processMutation.mutate(request);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium text-gray-900">Edit Text Content</h3>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowPreview(!showPreview)}
          >
            {showPreview ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            {showPreview ? 'Hide' : 'Show'} Preview
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleReset}
          >
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset
          </Button>
        </div>
      </div>

      <div className={`grid gap-6 ${showPreview ? 'lg:grid-cols-2' : 'grid-cols-1'}`}>
        {/* Text Editor */}
        <div className="space-y-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h4 className="text-sm font-medium text-blue-800 mb-2">
              Replace Text Placeholders
            </h4>
            <p className="text-xs text-blue-700">
              Customize the content by replacing each placeholder with your desired text.
            </p>
          </div>

          <div className="space-y-4 max-h-96 overflow-y-auto">
            {editablePlaceholders.map((placeholder, index) => (
              <div key={index} className="space-y-2">
                <label
                  htmlFor={`placeholder-${index}`}
                  className="block text-sm font-medium text-gray-700"
                >
                  <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-blue-100 text-blue-800 mr-2">
                    {placeholder}
                  </span>
                  Edit Content
                </label>
                <textarea
                  id={`placeholder-${index}`}
                  value={replacements[placeholder] || ''}
                  onChange={(e) => handleReplacementChange(placeholder, e.target.value)}
                  placeholder={`Enter content for ${placeholder}`}
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-200 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                />
              </div>
            ))}
          </div>

          {editablePlaceholders.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              <p>No editable content found in this worksheet.</p>
              <p className="text-sm">The template fields are automatically populated.</p>
            </div>
          )}
        </div>

        {/* Live Preview */}
        {showPreview && (
          <div className="space-y-4">
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <h4 className="text-sm font-medium text-gray-800 mb-2">
                Live Preview
              </h4>
              <p className="text-xs text-gray-600">
                See how your changes look in real-time.
              </p>
            </div>

            <div className="border border-gray-200 rounded-lg p-4 bg-white overflow-auto max-h-96">
              <div className="flex items-center justify-center min-h-[300px]">
                <div
                  className="svg-container"
                  style={{
                    transform: 'scale(0.6)',
                    transformOrigin: 'center top',
                    transition: 'transform 0.2s ease',
                    maxWidth: '100%'
                  }}
                >
                  {/* Use same inline SVG approach as SVGPreview for consistency */}
                  <div
                    className="inline-block shadow-lg rounded-lg overflow-hidden bg-white"
                    style={{ maxWidth: '100%', height: 'auto' }}
                  >
                    <svg
                      style={{ display: 'block', maxWidth: '100%', height: 'auto' }}
                      dangerouslySetInnerHTML={{ __html: processedSVG.replace('<svg', '<g').replace('</svg>', '</g>') }}
                      viewBox="0 0 800 1000"
                      width="800"
                      height="1000"
                      xmlns="http://www.w3.org/2000/svg"
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex justify-between pt-4 border-t border-gray-200">
        <Button variant="outline" onClick={onCancel}>
          <X className="mr-2 h-4 w-4" />
          Cancel
        </Button>
        
        <Button
          onClick={handleSave}
          disabled={processMutation.isLoading}
        >
          {processMutation.isLoading ? (
            <>
              <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-white"></div>
              Processing...
            </>
          ) : (
            <>
              <Save className="mr-2 h-4 w-4" />
              Save Changes
            </>
          )}
        </Button>
      </div>

      {/* Usage Tips */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-medium text-gray-900 mb-2">Tips</h4>
        <ul className="text-sm text-gray-600 space-y-1">
          <li className="flex items-start">
            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
            Keep text concise to fit within the design layout
          </li>
          <li className="flex items-start">
            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
            Use the preview to check how changes affect the overall design
          </li>
          <li className="flex items-start">
            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
            Leave fields empty to remove text placeholders entirely
          </li>
        </ul>
      </div>
    </div>
  );
};