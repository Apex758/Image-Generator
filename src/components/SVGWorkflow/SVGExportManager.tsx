import React, { useState } from 'react';
import { useMutation } from 'react-query';
import { Button } from '../ui/Button';
import { Download, FileText, FileImage, ArrowLeft, CheckCircle } from 'lucide-react';
import { svgApi, ExportSVGRequest } from '../../lib/api';

interface SVGExportManagerProps {
  svgContent: string;
  onBack: () => void;
}

interface ExportFormat {
  id: 'pdf' | 'docx' | 'png';
  name: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  useCase: string;
}

const exportFormats: ExportFormat[] = [
  {
    id: 'pdf',
    name: 'PDF Document',
    description: 'Ready for printing and sharing',
    icon: <FileText className="h-6 w-6" />,
    color: 'bg-red-50 border-red-200 text-red-700',
    useCase: 'Perfect for printing handouts and homework assignments'
  },
  {
    id: 'docx',
    name: 'Word Document',
    description: 'Editable in Microsoft Word',
    icon: <FileText className="h-6 w-6" />,
    color: 'bg-blue-50 border-blue-200 text-blue-700',
    useCase: 'Great for further editing and collaboration'
  },
  {
    id: 'png',
    name: 'PNG Image',
    description: 'High-quality image format',
    icon: <FileImage className="h-6 w-6" />,
    color: 'bg-green-50 border-green-200 text-green-700',
    useCase: 'Ideal for digital presentations and online use'
  }
];

export const SVGExportManager: React.FC<SVGExportManagerProps> = ({
  svgContent,
  onBack
}) => {
  const [selectedFormat, setSelectedFormat] = useState<'pdf' | 'docx' | 'png' | ''>('');
  const [filename, setFilename] = useState('worksheet');
  const [exportedFiles, setExportedFiles] = useState<string[]>([]);

  const exportMutation = useMutation(
    (request: ExportSVGRequest) => svgApi.export(request),
    {
      onSuccess: (data) => {
        // Create a temporary link to download the file
        const link = document.createElement('a');
        link.href = data.download_url;
        link.download = data.filename;
        link.click();
        
        // Track exported files
        setExportedFiles(prev => [...prev, data.filename]);
      },
      onError: (error: unknown) => {
        console.error('Export failed:', error);
        alert('Failed to export file. Please try again.');
      },
    }
  );

  const handleExport = () => {
    if (!selectedFormat) {
      alert('Please select an export format');
      return;
    }

    const request: ExportSVGRequest = {
      svg_content: svgContent,
      format: selectedFormat,
      filename: filename || 'worksheet'
    };

    exportMutation.mutate(request);
  };

  const getFileExtension = (format: string) => {
    switch (format) {
      case 'pdf': return '.pdf';
      case 'docx': return '.docx';
      case 'png': return '.png';
      default: return '';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium text-gray-900">Export Your Worksheet</h3>
        <Button variant="outline" onClick={onBack}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back
        </Button>
      </div>

      {/* Export Progress */}
      {exportedFiles.length > 0 && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-start">
            <CheckCircle className="h-5 w-5 text-green-600 mt-0.5 mr-3 flex-shrink-0" />
            <div>
              <h4 className="text-sm font-medium text-green-800 mb-1">
                Successfully Exported ({exportedFiles.length})
              </h4>
              <ul className="text-sm text-green-700 space-y-1">
                {exportedFiles.map((file, index) => (
                  <li key={index} className="flex items-center">
                    <span className="w-1 h-1 bg-green-600 rounded-full mr-2"></span>
                    {file}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Format Selection */}
      <div className="space-y-4">
        <div>
          <h4 className="text-sm font-medium text-gray-900 mb-3">Choose Export Format</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {exportFormats.map((format) => (
              <div
                key={format.id}
                className={`
                  relative p-4 border-2 rounded-lg cursor-pointer transition-all duration-200
                  ${selectedFormat === format.id 
                    ? `${format.color} border-opacity-100 shadow-md` 
                    : 'bg-white border-gray-200 hover:border-gray-300 hover:shadow-sm'
                  }
                `}
                onClick={() => setSelectedFormat(format.id)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    setSelectedFormat(format.id);
                  }
                }}
                aria-label={`Select ${format.name} format`}
              >
                {/* Selection indicator */}
                {selectedFormat === format.id && (
                  <div className="absolute top-2 right-2">
                    <div className="h-3 w-3 bg-current rounded-full"></div>
                  </div>
                )}

                <div className="flex items-start gap-3">
                  <div className={`
                    p-2 rounded-lg 
                    ${selectedFormat === format.id ? 'bg-white/50' : format.color}
                  `}>
                    {format.icon}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <h5 className="font-medium text-gray-900 mb-1">{format.name}</h5>
                    <p className="text-sm text-gray-600 mb-2">{format.description}</p>
                    <p className="text-xs text-gray-500">{format.useCase}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Filename Input */}
        <div>
          <label htmlFor="filename" className="block text-sm font-medium text-gray-700 mb-2">
            File Name
          </label>
          <div className="flex items-center gap-2">
            <input
              type="text"
              id="filename"
              value={filename}
              onChange={(e) => setFilename(e.target.value.replace(/[^a-zA-Z0-9-_]/g, ''))}
              placeholder="Enter filename (without extension)"
              className="flex-1 px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            {selectedFormat && (
              <span className="text-sm text-gray-500 font-mono">
                {getFileExtension(selectedFormat)}
              </span>
            )}
          </div>
          <p className="text-xs text-gray-500 mt-1">
            Only letters, numbers, hyphens, and underscores are allowed
          </p>
        </div>
      </div>

      {/* Preview */}
      <div className="space-y-4">
        <h4 className="text-sm font-medium text-gray-900">Final Preview</h4>
        <div className="border border-gray-200 rounded-lg p-4 bg-gray-50 overflow-auto max-h-96">
          <div className="flex items-center justify-center min-h-[400px]">
            <div
              className="svg-container"
              style={{
                transform: 'scale(0.4)', // Match live preview scale
                transformOrigin: 'center top',
                maxWidth: '100%'
              }}
            >
              <div
                className="shadow-sm rounded overflow-hidden bg-white"
                dangerouslySetInnerHTML={{ __html: svgContent }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Export Button */}
      <div className="flex justify-center pt-4">
        <Button
          onClick={handleExport}
          disabled={!selectedFormat || !filename.trim() || exportMutation.isLoading}
          size="lg"
          className="px-8"
        >
          {exportMutation.isLoading ? (
            <>
              <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-white"></div>
              Exporting...
            </>
          ) : (
            <>
              <Download className="mr-2 h-5 w-5" />
              Export {selectedFormat ? exportFormats.find(f => f.id === selectedFormat)?.name : 'Worksheet'}
            </>
          )}
        </Button>
      </div>

      {/* Usage Instructions */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h4 className="text-sm font-medium text-blue-800 mb-2">Usage Tips</h4>
        <ul className="text-sm text-blue-700 space-y-1">
          <li className="flex items-start">
            <span className="w-1.5 h-1.5 bg-blue-600 rounded-full mt-2 mr-2 flex-shrink-0"></span>
            PDF format is best for printing and ensuring consistent formatting across devices
          </li>
          <li className="flex items-start">
            <span className="w-1.5 h-1.5 bg-blue-600 rounded-full mt-2 mr-2 flex-shrink-0"></span>
            DOCX format allows you to make further edits in Microsoft Word
          </li>
          <li className="flex items-start">
            <span className="w-1.5 h-1.5 bg-blue-600 rounded-full mt-2 mr-2 flex-shrink-0"></span>
            PNG format is perfect for inserting into presentations or digital platforms
          </li>
          <li className="flex items-start">
            <span className="w-1.5 h-1.5 bg-blue-600 rounded-full mt-2 mr-2 flex-shrink-0"></span>
            You can export the same worksheet in multiple formats if needed
          </li>
        </ul>
      </div>
    </div>
  );
};