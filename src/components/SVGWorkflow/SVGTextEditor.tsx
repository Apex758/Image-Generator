import React, { useState, useEffect } from 'react';
import { useMutation } from 'react-query';
import { Button } from '../ui/Button';
import { Save, X, RotateCcw, Eye, EyeOff } from 'lucide-react';
import { svgApi, ProcessSVGRequest } from '../../lib/api';

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
  const [replacements, setReplacements] = useState<Record<string, string>>({});
  const [showPreview, setShowPreview] = useState(true);
  const [previewContent, setPreviewContent] = useState(svgContent);

  // Initialize replacements with placeholder names as default values
  useEffect(() => {
    const initialReplacements: Record<string, string> = {};
    placeholders.forEach(placeholder => {
      initialReplacements[placeholder] = placeholder;
    });
    setReplacements(initialReplacements);
  }, [placeholders]);

  // Update preview when replacements change
  useEffect(() => {
    let updatedContent = svgContent;
    Object.entries(replacements).forEach(([placeholder, replacement]) => {
      const regex = new RegExp(`{${placeholder}}`, 'g');
      updatedContent = updatedContent.replace(regex, replacement || placeholder);
    });
    setPreviewContent(updatedContent);
  }, [svgContent, replacements]);

  const processMutation = useMutation(
    (request: ProcessSVGRequest) => svgApi.process(request),
    {
      onSuccess: (data) => {
        onSave(data.svg_content);
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
    placeholders.forEach(placeholder => {
      initialReplacements[placeholder] = placeholder;
    });
    setReplacements(initialReplacements);
  };

  const handleSave = () => {
    const request: ProcessSVGRequest = {
      svg_content: svgContent,
      replacements
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
            {placeholders.map((placeholder, index) => (
              <div key={index} className="space-y-2">
                <label 
                  htmlFor={`placeholder-${index}`}
                  className="block text-sm font-medium text-gray-700"
                >
                  <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-gray-100 text-gray-800 mr-2">
                    {placeholder}
                  </span>
                  Replacement Text
                </label>
                <textarea
                  id={`placeholder-${index}`}
                  value={replacements[placeholder] || ''}
                  onChange={(e) => handleReplacementChange(placeholder, e.target.value)}
                  placeholder={`Enter replacement for ${placeholder}`}
                  rows={2}
                  className="w-full px-3 py-2 border border-gray-200 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                />
              </div>
            ))}
          </div>

          {placeholders.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              <p>No text placeholders found in this SVG.</p>
              <p className="text-sm">You can proceed directly to export.</p>
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
              <div
                className="flex items-center justify-center"
                style={{ maxWidth: '100%' }}
              >
                <div
                  className="shadow-sm rounded overflow-hidden bg-white max-w-full"
                  style={{ transform: 'scale(0.8)', transformOrigin: 'center' }}
                  dangerouslySetInnerHTML={{ __html: previewContent }}
                />
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