import React, { useState, useMemo } from 'react';
import { Button } from '../ui/Button';
import { Edit, Download, ZoomIn, ZoomOut } from 'lucide-react';

interface SVGPreviewProps {
  svgContent: string;
  placeholders: string[];
  onEditText: () => void;
  onExport: () => void;
}

// Function to safely parse viewBox attribute
const getSafeViewBox = (svgString: string): string => {
  const viewBoxMatch = svgString.match(/viewBox="([^"]+)"/);
  if (viewBoxMatch && viewBoxMatch[1]) {
    const values = viewBoxMatch[1].split(/\s+/).map(v => parseFloat(v) || 0);
    return values.join(' ');
  }
  return '0 0 800 600'; // Fallback
};

export const SVGPreview: React.FC<SVGPreviewProps> = ({
  svgContent,
  placeholders,
  onEditText,
  onExport
}) => {
  const [scale, setScale] = useState(1);
  const [showPlaceholders, setShowPlaceholders] = useState(true);

  const zoomIn = () => setScale(prev => Math.min(prev + 0.2, 2));
  const zoomOut = () => setScale(prev => Math.max(prev - 0.2, 0.5));

  // Memoize the SVG processing to avoid re-running on every render
  const processedSVG = useMemo(() => {
    let content = svgContent;

    // Sanitize viewBox
    const safeViewBox = getSafeViewBox(content);
    content = content.replace(/viewBox="([^"]+)"/, `viewBox="${safeViewBox}"`);

    // Highlight placeholders
    if (showPlaceholders) {
      content = content.replace(
        /\{([^}]+)\}/g,
        '<tspan fill="#ef4444" stroke="#ef4444" stroke-width="0.5">{$1}</tspan>'
      );
    }
    return content;
  }, [svgContent, showPlaceholders]);

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-4">Preview Your Worksheet</h3>
        
        {/* Controls */}
        <div className="flex items-center justify-between mb-4 p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={zoomOut}
                disabled={scale <= 0.5}
              >
                <ZoomOut className="h-4 w-4" />
              </Button>
              <span className="text-sm text-gray-600 min-w-[4rem] text-center">
                {Math.round(scale * 100)}%
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={zoomIn}
                disabled={scale >= 2}
              >
                <ZoomIn className="h-4 w-4" />
              </Button>
            </div>
            
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="showPlaceholders"
                checked={showPlaceholders}
                onChange={(e) => setShowPlaceholders(e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <label htmlFor="showPlaceholders" className="text-sm text-gray-700">
                Highlight text placeholders
              </label>
            </div>
          </div>

          <div className="flex gap-2">
            <Button variant="outline" onClick={onEditText}>
              <Edit className="mr-2 h-4 w-4" />
              Edit Text
            </Button>
            <Button onClick={onExport}>
              <Download className="mr-2 h-4 w-4" />
              Export
            </Button>
          </div>
        </div>

        {/* Placeholders Info */}
        {placeholders.length > 0 && (
          <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h4 className="text-sm font-medium text-blue-800 mb-2">
              Text Placeholders Found ({placeholders.length})
            </h4>
            <div className="flex flex-wrap gap-2">
              {placeholders.map((placeholder, index) => (
                <span
                  key={index}
                  className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                >
                  {placeholder}
                </span>
              ))}
            </div>
            <p className="text-xs text-blue-700 mt-2">
              Click "Edit Text" to customize these placeholders with your own content.
            </p>
          </div>
        )}
      </div>

      {/* SVG Preview */}
      <div className="border border-gray-200 rounded-lg p-4 bg-white overflow-auto">
        <div 
          className="flex items-center justify-center min-h-[400px]"
          style={{
            transform: `scale(${scale})`,
            transformOrigin: 'center',
            transition: 'transform 0.2s ease'
          }}
        >
          <div
            className="shadow-lg rounded-lg overflow-hidden bg-white"
            dangerouslySetInnerHTML={{ __html: processedSVG }}
          />
        </div>
      </div>

      {/* Instructions */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-medium text-gray-900 mb-2">Next Steps</h4>
        <ul className="text-sm text-gray-600 space-y-1">
          <li className="flex items-start">
            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
            Review the generated worksheet preview above
          </li>
          <li className="flex items-start">
            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
            Click "Edit Text" to customize placeholder content
          </li>
          <li className="flex items-start">
            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
            Use zoom controls to inspect details
          </li>
          <li className="flex items-start">
            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
            Click "Export" when ready to download your worksheet
          </li>
        </ul>
      </div>
    </div>
  );
};