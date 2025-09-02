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

  // FIXED: Extract actual content from the filled SVG, not just placeholders
  const extractCoreContentFromSVG = useCallback((): EditableContent[] => {
    const extractedContent: EditableContent[] = [];

    try {
      // Parse the SVG to extract actual text content
      const parser = new DOMParser();
      const doc = parser.parseFromString(svgContent, 'image/svg+xml');

      // Extract from foreignObject elements (these contain the actual AI-generated content)
      const foreignObjects = doc.querySelectorAll('foreignObject div');
      foreignObjects.forEach((element, index) => {
        const textContent = element.textContent?.trim();
        if (textContent && textContent.length > 0 && !textContent.includes('Name:') && !textContent.includes('Date:')) {
          
          // Identify question content
          if (textContent.match(/^\d+\./)) {
            const questionNum = textContent.match(/^(\d+)\./)?.[1] || (index + 1).toString();
            extractedContent.push({
              id: `question_${questionNum}`,
              label: `Question ${questionNum}`,
              value: textContent,
              type: 'textarea',
              category: 'questions'
            });
          }
          // Identify instructions
          else if (textContent.toLowerCase().includes('look at') || 
                   textContent.toLowerCase().includes('fill in') || 
                   textContent.toLowerCase().includes('answer') ||
                   textContent.length > 50) {
            extractedContent.push({
              id: `instructions`,
              label: 'Main Instructions',
              value: textContent,
              type: 'textarea',
              category: 'content'
            });
          }
        }
      });

      // Extract from regular text elements
      const textElements = doc.querySelectorAll('text, tspan');
      textElements.forEach((element) => {
        const textContent = element.textContent?.trim();
        if (textContent && textContent.length > 0) {
          
          // Extract header info (subject, grade, topic)
          if (textContent.includes(' - Grade ')) {
            const parts = textContent.split(' - Grade ');
            if (parts.length === 2) {
              extractedContent.push({
                id: 'subject',
                label: 'Subject',
                value: parts[0].trim(),
                type: 'text',
                category: 'header'
              });
              extractedContent.push({
                id: 'grade',
                label: 'Grade Level', 
                value: parts[1].trim(),
                type: 'text',
                category: 'header'
              });
            }
          }
          // Extract topic
          else if (textContent.startsWith('Topic: ')) {
            extractedContent.push({
              id: 'topic',
              label: 'Topic',
              value: textContent.replace('Topic: ', '').trim(),
              type: 'text',
              category: 'header'
            });
          }
        }
      });

      // Add missing basic fields if not extracted
      const requiredFields = [
        { id: 'subject', label: 'Subject', category: 'header' as const, value: 'Science' },
        { id: 'grade', label: 'Grade Level', category: 'header' as const, value: 'Grade 5' },
        { id: 'topic', label: 'Topic', category: 'header' as const, value: 'water cycle' },
        { id: 'instructions', label: 'Main Instructions', category: 'content' as const, value: 'Look at the image showing the water cycle and fill in the blanks with the correct words from the word bank provided.' }
      ];

      requiredFields.forEach(field => {
        if (!extractedContent.find(item => item.id === field.id)) {
          extractedContent.push({
            id: field.id,
            label: field.label,
            value: field.value,
            type: field.id === 'instructions' ? 'textarea' : 'text',
            category: field.category
          });
        }
      });

      // Ensure we have entries for all 10 questions
      for (let i = 1; i <= 10; i++) {
        if (!extractedContent.find(item => item.id === `question_${i}`)) {
          extractedContent.push({
            id: `question_${i}`,
            label: `Question ${i}`,
            value: `${i}. [Enter question ${i} content here]`,
            type: 'textarea',
            category: 'questions'
          });
        }
      }

    } catch (error) {
      console.warn('Failed to extract content from SVG:', error);
      
      // Fallback to placeholder-based extraction
      placeholders.forEach((placeholder) => {
        let category: EditableContent['category'] = 'other';
        let label = placeholder.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase());
        
        if (placeholder.includes('subject') || placeholder.includes('grade') || placeholder.includes('topic')) {
          category = 'header';
        } else if (placeholder.includes('instruction')) {
          category = 'content';
          label = 'Instructions';
        } else if (placeholder.includes('question')) {
          category = 'questions';
          const questionNum = placeholder.match(/\d+/)?.[0] || '';
          label = `Question ${questionNum}`;
        }

        extractedContent.push({
          id: `placeholder_${placeholder}`,
          label,
          value: `[${placeholder}]`,
          type: placeholder.includes('question') || placeholder.includes('instruction') ? 'textarea' : 'text',
          category
        });
      });
    }

    return extractedContent.sort((a, b) => {
      const categoryOrder = ['header', 'content', 'questions', 'wordbank', 'other'];
      const catCompare = categoryOrder.indexOf(a.category) - categoryOrder.indexOf(b.category);
      
      // Within questions category, sort by question number
      if (catCompare === 0 && a.category === 'questions') {
        const aNum = parseInt(a.id.replace('question_', '')) || 0;
        const bNum = parseInt(b.id.replace('question_', '')) || 0;
        return aNum - bNum;
      }
      
      return catCompare;
    });

  }, [svgContent, placeholders]);

  // Extract words from fill-in-the-blank questions for word bank
  const extractWordsForWordBank = useCallback((content: EditableContent[]): string[] => {
    // Extract from the actual question content
    const words: string[] = [];
    
    content.forEach(item => {
      if (item.category === 'questions' && item.value.includes('_____')) {
        // Extract context words that could be answers
        const questionWords = item.value
          .toLowerCase()
          .replace(/\d+\./g, '')
          .replace(/_____/g, '')
          .split(/\s+/)
          .filter(word => word.length > 3 && /^[a-zA-Z]+$/.test(word))
          .slice(0, 1); // Take 1 context word per question
        
        words.push(...questionWords);
      }
    });

    // Add common water cycle words to complete the word bank
    const commonWords = ['water', 'evaporates', 'condenses', 'precipitation', 'rain', 'runoff', 'groundwater', 'sustains', 'returns', 'hydrologic'];
    
    // Combine and deduplicate
    const allWords = [...new Set([...words, ...commonWords])];
    return allWords.slice(0, 10); // Max 10 words
  }, []);

  // Initialize content extraction
  useEffect(() => {
    const extracted = extractCoreContentFromSVG();
    setEditableContent(extracted);
    
    // Extract word bank words
    const words = extractWordsForWordBank(extracted);
    setWordBankWords(words);
  }, [extractCoreContentFromSVG, extractWordsForWordBank]);

  // Store original content values for replacement tracking
  const [originalContentMap, setOriginalContentMap] = useState<Record<string, string>>({});
  const [lastUpdateTime] = useState(Date.now()); // Track updates for debugging

  // FIXED: Real-time preview update that actually works
  const updatePreview = useCallback(() => {
    let updatedContent = svgContent;
    
    // Replace content based on editable items - handle both placeholder patterns AND actual content
    editableContent.forEach(item => {
      if (item.value && item.value.trim()) {
        const cleanValue = item.value.trim();
        const originalValue = originalContentMap[item.id];
        
        // Try multiple replacement strategies
        if (item.id === 'subject') {
          // Replace both placeholder and actual content
          updatedContent = updatedContent.replace(/\[subject\]/gi, cleanValue);
          if (originalValue) {
            updatedContent = updatedContent.replace(new RegExp(originalValue.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'), cleanValue);
          }
          // Also try common patterns
          updatedContent = updatedContent.replace(/Science - Grade/g, `${cleanValue} - Grade`);
          updatedContent = updatedContent.replace(/Science<\/text>/g, `${cleanValue}</text>`);
        }
        else if (item.id === 'grade') {
          updatedContent = updatedContent.replace(/\[grade\]/gi, cleanValue);
          updatedContent = updatedContent.replace(/Grade \d+/g, cleanValue);
          updatedContent = updatedContent.replace(/Grade Grade \d+/g, `Grade ${cleanValue}`);
          if (originalValue) {
            updatedContent = updatedContent.replace(new RegExp(originalValue.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'), cleanValue);
          }
        }
        else if (item.id === 'topic') {
          updatedContent = updatedContent.replace(/\[topic\]/gi, cleanValue);
          updatedContent = updatedContent.replace(/Topic: [^<]*/g, `Topic: ${cleanValue}`);
          if (originalValue) {
            updatedContent = updatedContent.replace(new RegExp(originalValue.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'), cleanValue);
          }
        }
        else if (item.id === 'instructions') {
          updatedContent = updatedContent.replace(/\[instructions\]/gi, cleanValue);
          // Replace in foreignObject div content
          const instructionRegex = /<div[^>]*>(.*?)<\/div>/gs;
          updatedContent = updatedContent.replace(instructionRegex, (match, content) => {
            if (content.includes('Look at') || content.includes('Fill in') || content.includes('word bank')) {
              return match.replace(content, cleanValue);
            }
            return match;
          });
        }
        else if (item.id.startsWith('question_')) {
          const questionNum = item.id.replace('question_', '');
          updatedContent = updatedContent.replace(new RegExp(`\\[question${questionNum}\\]`, 'gi'), cleanValue);
          
          // Replace actual question content in foreignObject divs
          const questionRegex = new RegExp(`<div[^>]*>\\s*${questionNum}\\.[^<]*</div>`, 'gs');
          updatedContent = updatedContent.replace(questionRegex, (match) => {
            return match.replace(/>\s*\d+\.[^<]*</, `>${cleanValue}<`);
          });
        }
      }
    });

    // Handle word bank
    if (includeWordBank && wordBankWords.length > 0) {
      // Remove existing word bank
      updatedContent = updatedContent.replace(/<!-- Word Bank Section -->[\s\S]*?(?=<!-- (?:Activity|Footer)|$)/g, '');
      
      const wordBankSVG = generateWordBankSVG(wordBankWords);
      const insertPoint = updatedContent.indexOf('<!-- Activity Section -->') !== -1
        ? '<!-- Activity Section -->'
        : '<!-- Footer -->';
      updatedContent = updatedContent.replace(
        insertPoint,
        `${wordBankSVG}\n  ${insertPoint}`
      );
    }
    
    setPreviewContent(updatedContent);
  }, [svgContent, editableContent, includeWordBank, wordBankWords, originalContentMap]);

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
    // No need for immediate update here - useEffect will handle it
  };

  const handleWordBankChange = (index: number, value: string) => {
    setWordBankWords(prev => {
      const newWords = [...prev];
      newWords[index] = value;
      return newWords;
    });
    // No need for immediate update here - useEffect will handle it
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
    
    // Reset original content map
    const originalMap: Record<string, string> = {};
    extracted.forEach(item => {
      originalMap[item.id] = item.value;
    });
    setOriginalContentMap(originalMap);
    
    const words = extractWordsForWordBank(extracted);
    setWordBankWords(words);
    
    // The useEffect will handle the preview update
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

  // Memoize the SVG processing - re-process when preview content changes
  const processedSVG = useMemo(() => {
    const content = cleanSVGContent(previewContent);
    return content;
  }, [previewContent]); // This will trigger re-processing when previewContent changes

  const handleSave = () => {
    // Create text replacements object with improved mapping
    const textReplacements: Record<string, string> = {};
    
    editableContent.forEach(item => {
      if (item.value && item.value.trim()) {
        const cleanValue = item.value.trim();
        
        // Map content to proper replacement keys
        if (item.id === 'subject') {
          textReplacements['subject'] = cleanValue;
        } else if (item.id === 'grade') {
          textReplacements['grade'] = cleanValue;
        } else if (item.id === 'topic') {
          textReplacements['topic'] = cleanValue;
        } else if (item.id === 'instructions') {
          textReplacements['instructions'] = cleanValue;
        } else if (item.id.startsWith('question_')) {
          const questionNum = item.id.replace('question_', '');
          textReplacements[`question${questionNum}`] = cleanValue;
        }
      }
    });

    // Add word bank words to replacements
    wordBankWords.forEach((word, index) => {
      if (word && word.trim()) {
        textReplacements[`word_bank_word${index + 1}`] = word.trim();
      }
    });

    const request: ProcessSVGRequest = {
      svg_content: previewContent,
      text_replacements: textReplacements,
      add_writing_lines: false
    };
    
    processMutation.mutate(request);
  };

  // Group content by category (exclude wordbank category from editing sections)
  const contentByCategory = editableContent.reduce((acc, item) => {
    if (item.category !== 'wordbank') {  // Don't show wordbank fields in editing sections
      if (!acc[item.category]) acc[item.category] = [];
      acc[item.category].push(item);
    }
    return acc;
  }, {} as Record<string, EditableContent[]>);

  const categoryLabels = {
    header: 'Header Information',
    content: 'Instructions & Content', 
    questions: 'Questions',
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
                <span className="text-xs text-green-600 ml-2">
                  (Last updated: {new Date(lastUpdateTime).toLocaleTimeString()})
                </span>
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
                    key={previewContent.length} // Force re-render when content changes
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
        <h4 className="text-sm font-medium text-gray-900 mb-2">💡 Enhanced Editor Features</h4>
        <ul className="text-sm text-gray-600 space-y-1">
          <li className="flex items-start">
            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
            <strong>Live Preview:</strong> See your changes instantly as you type
          </li>
          <li className="flex items-start">
            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
            <strong>Smart Content Recognition:</strong> Automatically extracts actual AI-generated content
          </li>
          <li className="flex items-start">
            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
            <strong>All 10 Questions:</strong> Edit all questions including those that may overflow to page 2
          </li>
          <li className="flex items-start">
            <span className="w-1.5 h-1.5 bg-gray-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
            <strong>Word Bank Management:</strong> Edit word bank separately at the top, positioned after questions
          </li>
        </ul>
      </div>
    </div>
  );
};