import React, { useState, useMemo } from 'react';
import { Button } from '../ui/Button';
import { Edit, Download, ZoomIn, ZoomOut } from 'lucide-react';

interface SVGPreviewProps {
  svgContent: string;
  placeholders: string[];
  onEditText: () => void;
  onExport: () => void;
}

// Function to safely clean and prepare SVG content for rendering
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

export const SVGPreview: React.FC<SVGPreviewProps> = ({
  svgContent,
  placeholders,
  onEditText,
  onExport
}) => {
  const [scale, setScale] = useState(0.8); // Start with smaller scale to fit better
  const [showPlaceholders, setShowPlaceholders] = useState(true);

  const zoomIn = () => setScale(prev => Math.min(prev + 0.2, 2));
  const zoomOut = () => setScale(prev => Math.max(prev - 0.2, 0.3));

  // Memoize the SVG processing to avoid re-running on every render
  const processedSVG = useMemo(() => {
    let content = cleanSVGContent(svgContent);

    // Highlight placeholders if enabled
    if (showPlaceholders) {
      // Look for placeholder text patterns and highlight them
      content = content.replace(
        /\[([^\]]+)\]/g,
        '<tspan fill="#ef4444" font-weight="bold">[$1]</tspan>'
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
                disabled={scale <= 0.3}
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
        <div className="flex items-center justify-center min-h-[400px]">
          <div 
            className="svg-container"
            style={{
              transform: `scale(${scale})`,
              transformOrigin: 'center top',
              transition: 'transform 0.2s ease',
              maxWidth: '100%'
            }}
          >
            {/* Use an inline SVG approach that's more reliable */}
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

      {/* Alternative rendering method as fallback */}
      <div className="border-t pt-4">
        <details className="mt-4">
          <summary className="text-sm font-medium text-gray-700 cursor-pointer hover:text-blue-600">
            Alternative View (if preview doesn't work)
          </summary>
          <div className="mt-2 border border-gray-200 rounded p-4 bg-gray-50 overflow-auto max-h-64">
            <iframe
              srcDoc={`
                <!DOCTYPE html>
                <html>
                <head>
                  <meta charset="UTF-8">
                  <style>
                    body { margin: 0; padding: 20px; background: white; }
                    svg { max-width: 100%; height: auto; }
                  </style>
                </head>
                <body>
                  ${processedSVG}
                </body>
                </html>
              `}
              width="100%"
              height="300"
              style={{ border: 'none', background: 'white' }}
              title="SVG Preview"
            />
          </div>
        </details>
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