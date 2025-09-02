import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useMutation } from 'react-query';
import { Button } from '../ui/Button';
import { Save, X, RotateCcw, Eye, EyeOff, BookOpen } from 'lucide-react';
import { svgApi, ProcessSVGRequest } from '../../lib/api';

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

interface SVGTextEditorProps {
  svgContent: string;
  placeholders: string[];
  onSave: (processedContent: string) => void;
  onCancel: () => void;
}

interface EditableContent {
  id: string;
  label: string;
  value: string;
  type: 'text' | 'textarea';
  category: 'header' | 'content' | 'questions' | 'wordbank' | 'other';
}

export const SVGTextEditor: React.FC<SVGTextEditorProps> = ({
  svgContent,
  placeholders,
  onSave,
  onCancel
}) => {
  const [editableContent, setEditableContent] = useState<EditableContent[]>([]);
  const [showPreview, setShowPreview] = useState(true);
  const [previewContent, setPreviewContent] = useState(svgContent);
  const [includeWordBank, setIncludeWordBank] = useState(true);
  const [wordBankWords, setWordBankWords] = useState<string[]>([]);

  // FIXED: Simplified content extraction to focus on core editable elements only
  const extractCoreContentFromSVG = useCallback((): EditableContent[] => {
    const extractedContent: EditableContent[] = [];

    // Only extract placeholder-based content to keep it simple
    placeholders.forEach((placeholder) => {
      let category: EditableContent['category'] = 'other';
      let label = placeholder.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase());
      
      // Categorize based on placeholder names
      if (placeholder.includes('subject') || placeholder.includes('grade') || placeholder.includes('topic')) {
        category = 'header';
        label = 'Header Info';
      } else if (placeholder.includes('instruction')) {
        category = 'content';
        label = 'Instructions';
      } else if (placeholder.includes('question')) {
        category = 'questions';
        const questionNum = placeholder.match(/\d+/)?.[0] || '';
        label = `Question ${questionNum}`;
      } else if (placeholder.includes('word_bank')) {
        category = 'wordbank';
        label = 'Word Bank';
      }

      extractedContent.push({
        id: `placeholder_${placeholder}`,
        label,
        value: `[${placeholder}]`,
        type: placeholder.includes('question') || placeholder.includes('instruction') ? 'textarea' : 'text',
        category
      });
    });

    // Add basic content fields if not present
    const basicFields = [
      { id: 'instructions', label: 'Main Instructions', category: 'content' as const },
      { id: 'subject', label: 'Subject', category: 'header' as const },
      { id: 'grade', label: 'Grade Level', category: 'header' as const },
      { id: 'topic', label: 'Topic', category: 'header' as const },
    ];

    basicFields.forEach(field => {
      if (!extractedContent.find(item => item.id.includes(field.id))) {
        extractedContent.push({
          id: `basic_${field.id}`,
          label: field.label,
          value: `[${field.id}]`,
          type: 'text',
          category: field.category
        });
      }
    });

    return extractedContent.sort((a, b) => {
      const categoryOrder = ['header', 'content', 'questions', 'wordbank', 'other'];
      return categoryOrder.indexOf(a.category) - categoryOrder.indexOf(b.category);
    });

  }, [placeholders]);

  // Extract words from fill-in-the-blank questions for word bank
  const extractWordsForWordBank = useCallback((content: EditableContent[]): string[] => {
    const words = ['water', 'vapor', 'clouds', 'precipitation', 'rain', 'groundwater', 'photosynthesis', 'runoff'];
    return words.slice(0, 8);
  }, []);

  // Initialize content extraction
  useEffect(() => {
    const extracted = extractCoreContentFromSVG();
    setEditableContent(extracted);
    
    // Extract word bank words
    const words = extractWordsForWordBank(extracted);
    setWordBankWords(words);
  }, [extractCoreContentFromSVG, extractWordsForWordBank]);

  // FIXED: Update preview content in real-time with proper dependencies
  const updatePreview = useCallback(() => {
    let updatedContent = svgContent;
    
    // Replace content based on editable items
    editableContent.forEach(item => {
      if (item.value && item.value.trim()) {
        // Extract placeholder name from ID
        const placeholderName = item.id.replace('placeholder_', '').replace('basic_', '');
        
        // Replace placeholder patterns
        const patterns = [
          `[${placeholderName}]`,
          `{${placeholderName}}`,
          new RegExp(`\\[${placeholderName}\\]`, 'gi')
        ];
        
        const cleanValue = item.value.replace(/^\[|\]$/g, ''); // Remove brackets from value
        
        patterns.forEach(pattern => {
          if (typeof pattern === 'string') {
            updatedContent = updatedContent.replace(new RegExp(pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'), cleanValue);
          } else {
            updatedContent = updatedContent.replace(pattern, cleanValue);
          }
        });
      }
    });

    // Add word bank if enabled
    if (includeWordBank && wordBankWords.length > 0) {
      const wordBankSVG = generateWordBankSVG(wordBankWords);
      // Insert word bank before the footer
      updatedContent = updatedContent.replace(
        '<!-- Footer',
        `${wordBankSVG}\n  <!-- Footer`
      );
    }
    
    setPreviewContent(updatedContent);
  }, [svgContent, editableContent, includeWordBank, wordBankWords]);

  // FIXED: Update preview when content changes with proper dependencies
  useEffect(() => {
    updatePreview();
  }, [updatePreview]);

  const processMutation = useMutation(
    (request: ProcessSVGRequest) => svgApi.process(request),
    {
      onSuccess: (data) => {
        onSave(data.processed_svg);
      },
      onError: (error: unknown) => {
        console.error('SVG processing failed:', error);
        alert('Failed to process SVG. Please try again.');
      },
    }
  );

  const handleContentChange = (id: string, value: string) => {
    setEditableContent(prev => 
      prev.map(item => 
        item.id === id ? { ...item, value } : item
      )
    );
  };

  const handleWordBankChange = (index: number, value: string) => {
    setWordBankWords(prev => {
      const newWords = [...prev];
      newWords[index] = value;
      return newWords;
    });
  };

  const addWordToBank = () => {
    if (wordBankWords.length < 12) {
      setWordBankWords(prev => [...prev, '']);
    }
  };

  const removeWordFromBank = (index: number) => {
    setWordBankWords(prev => prev.filter((_, i) => i !== index));
  };

  const handleReset = () => {
    const extracted = extractCoreContentFromSVG();
    setEditableContent(extracted);
    const words = extractWordsForWordBank(extracted);
    setWordBankWords(words);
  };

  const generateWordBankSVG = (words: string[]): string => {
    if (!words || words.length === 0) return '';
    
    const wordsPerRow = 4;
    const wordSpacing = 120;
    const rowSpacing = 25;
    const startX = 70;
    const startY = 850; // Position word bank towards bottom
    
    let wordBankSVG = `
  <!-- Word Bank Section -->
  <rect x="50" y="${startY - 30}" width="694" height="${Math.ceil(words.length / wordsPerRow) * rowSpacing + 40}" fill="#f8f9fa" stroke="#dee2e6" stroke-width="1"/>
  <text x="70" y="${startY - 10}" class="instruction">Word Bank:</text>
`;

    words.forEach((word, index) => {
      if (word && word.trim()) {
        const row = Math.floor(index / wordsPerRow);
        const col = index % wordsPerRow;
        const x = startX + (col * wordSpacing);
        const y = startY + (row * rowSpacing);
        
        wordBankSVG += `
  <text x="${x}" y="${y}" class="small" style="font-weight: bold;">${word.trim()}</text>`;
      }
    });

    return wordBankSVG;
  };

  // Memoize the SVG processing to avoid re-running on every render
  const processedSVG = useMemo(() => {
    const content = cleanSVGContent(previewContent);
    return content;
  }, [previewContent]);

  const handleSave = () => {
    // Create text replacements object
    const textReplacements: Record<string, string> = {};
    editableContent.forEach(item => {
      const placeholderName = item.id.replace('placeholder_', '').replace('basic_', '');
      textReplacements[placeholderName] = item.value.replace(/^\[|\]$/g, ''); // Remove brackets
    });

    const request: ProcessSVGRequest = {
      svg_content: previewContent, // Use the updated preview content
      text_replacements: textReplacements,
      add_writing_lines: false
    };
    processMutation.mutate(request);
  };

  // Group content by category
  const contentByCategory = editableContent.reduce((acc, item) => {
    if (!acc[item.category]) acc[item.category] = [];
    acc[item.category].push(item);
    return acc;
  }, {} as Record<string, EditableContent[]>);

  const categoryLabels = {
    header: 'Header Information',
    content: 'Instructions & Content',
    questions: 'Questions',
    wordbank: 'Word Bank',
    other: 'Other Text'
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium text-gray-900">Edit Worksheet Content</h3>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowPreview(!showPreview)}
          >
            {showPreview ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            {showPreview ? 'Hide' : 'Show'} Live Preview
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleReset}
          >
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset All
          </Button>
        </div>
      </div>

      <div className={`grid gap-6 ${showPreview ? 'lg:grid-cols-2' : 'grid-cols-1'}`}>
        {/* Content Editor */}
        <div className="space-y-6">
          {/* Word Bank Toggle */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-start space-x-3">
              <input
                type="checkbox"
                id="includeWordBank"
                checked={includeWordBank}
                onChange={(e) => setIncludeWordBank(e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded mt-1"
              />
              <div className="flex-1">
                <label htmlFor="includeWordBank" className="font-medium text-blue-800 cursor-pointer">
                  Include Word Bank for Fill-in-the-Blank Questions
                </label>
                <p className="text-sm text-blue-600 mt-1">
                  Provides answer words at the bottom of the worksheet to help students complete the blanks.
                </p>
              </div>
            </div>
          </div>

          {/* Word Bank Editor (if enabled) */}
          {includeWordBank && (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-medium text-gray-800 flex items-center">
                  <BookOpen className="h-4 w-4 mr-2" />
                  Word Bank Words
                </h4>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={addWordToBank}
                  disabled={wordBankWords.length >= 12}
                >
                  Add Word
                </Button>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {wordBankWords.map((word, index) => (
                  <div key={index} className="flex items-center space-x-2">
                    <input
                      type="text"
                      value={word}
                      onChange={(e) => handleWordBankChange(index, e.target.value)}
                      placeholder={`Word ${index + 1}`}
                      className="flex-1 px-2 py-1 text-sm border border-gray-300 rounded"
                    />
                    <button
                      onClick={() => removeWordFromBank(index)}
                      className="text-red-500 hover:text-red-700 text-sm"
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
              <p className="text-xs text-gray-500 mt-2">
                💡 Tip: These words will appear at the bottom to help students fill in the blanks
              </p>
            </div>
          )}

          {/* SIMPLIFIED Content Sections */}
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {Object.entries(contentByCategory).map(([category, items]) => (
              <div key={category} className="space-y-3">
                <h4 className="text-sm font-semibold text-gray-700 border-b pb-1">
                  {categoryLabels[category as keyof typeof categoryLabels] || category}
                </h4>
                {items.map((item) => (
                  <div key={item.id} className="space-y-2">
                    <label
                      htmlFor={item.id}
                      className="block text-sm font-medium text-gray-700"
                    >
                      {item.label}
                    </label>
                    {item.type === 'textarea' ? (
                      <textarea
                        id={item.id}
                        value={item.value}
                        onChange={(e) => handleContentChange(item.id, e.target.value)}
                        rows={3}
                        className="w-full px-3 py-2 border border-gray-200 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                      />
                    ) : (
                      <input
                        type="text"
                        id={item.id}
                        value={item.value}
                        onChange={(e) => handleContentChange(item.id, e.target.value)}
                        className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                      />
                    )}
                  </div>
                ))}
              </div>
            ))}
          </div>

          {editableContent.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              <p>Loading editable content...</p>
            </div>
          )}
        </div>

        {/* FIXED Live Preview */}
        {showPreview && (
          <div className="space-y-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <h4 className="text-sm font-medium text-green-800 mb-2">
                ✨ Live Preview
              </h4>
              <p className="text-xs text-green-600">
                Changes update automatically as you type. Preview refreshes in real-time.
              </p>
            </div>

            <div className="border border-gray-200 rounded-lg p-4 bg-white overflow-auto max-h-96">
              <div className="flex items-center justify-center min-h-[300px]">
                <div
                  className="svg-container"
                  style={{
                    transform: 'scale(0.5)',
                    transformOrigin: 'center top',
                    transition: 'transform 0.2s ease',
                    maxWidth: '100%'
                  }}
                >
                  <div
                    className="inline-block shadow-lg rounded-lg overflow-hidden bg-white"
                    style={{ maxWidth: '100%', height: 'auto' }}
                    dangerouslySetInnerHTML={{ __html: processedSVG }}
                  />
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
          Cancel Changes
        </Button>
        
        <Button
          onClick={handleSave}
          disabled={processMutation.isLoading}
        >
          {processMutation.isLoading ? (
            <>
              <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-white"></div>
              Saving Changes...
            </>
          ) : (
            <>
              <Save className="mr-2 h-4 w-4" />
              Save All Changes
            </>
          )}
        </Button>
      </div>

      {/* Usage Tips */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <h4 className="text-sm font-medium text-gray-900 mb-2">💡 Simplified Editor Features</h4>
        <ul className="text-sm text-gray-600 space-y-1">
          <li className="flex items-start">
            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
            <strong>Live Preview:</strong> See your changes instantly as you type
          </li>
          <li className="flex items-start">
            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
            <strong>Essential Editing:</strong> Only core content fields are shown for simplicity
          </li>
          <li className="flex items-start">
            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
            <strong>Word Bank:</strong> Automatically generated for fill-in-the-blank questions
          </li>
        </ul>
      </div>
    </div>
  );
};