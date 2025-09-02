import React, { useState } from 'react';
import { useMutation, useQueryClient } from 'react-query';
import { Button } from '../ui/Button';
import { Sparkles, Loader2, FileText } from 'lucide-react';
import { ContentTypeSelector, contentTypes } from './ContentTypeSelector';
import { SVGPreview } from './SVGPreview';
import { SVGTextEditor } from './SVGTextEditor';
import { SVGExportManager } from './SVGExportManager';
import { svgApi, GenerateSVGRequest, GenerateSVGResponse } from '../../lib/api';

export const SVGGenerator: React.FC = () => {
  const [currentStep, setCurrentStep] = useState<'select' | 'configure' | 'preview' | 'edit' | 'export'>('select');
  const [selectedContentType, setSelectedContentType] = useState<'image_comprehension' | 'comic' | 'math' | 'worksheet' | ''>('');
  const [subject, setSubject] = useState('');
  const [topic, setTopic] = useState('');
  const [gradeLevel, setGradeLevel] = useState('');
  const [layoutStyle, setLayoutStyle] = useState('layout1');
  const [numQuestions, setNumQuestions] = useState(5);
  const [questionTypes, setQuestionTypes] = useState(['fill_blank']);
  const [imageFormat, setImageFormat] = useState('landscape'); // Changed from aspectRatio
  const [imageCount, setImageCount] = useState(3);
  const [customInstructions, setCustomInstructions] = useState('');
  const [generatedSVG, setGeneratedSVG] = useState<GenerateSVGResponse | null>(null);
  const [processedSVG, setProcessedSVG] = useState<string>('');

  const queryClient = useQueryClient();

  const generateMutation = useMutation(
    (request: GenerateSVGRequest) => svgApi.generate(request),
    {
      onSuccess: (data) => {
        setGeneratedSVG(data);
        setCurrentStep('preview');
        queryClient.invalidateQueries('svg-items');
      },
      onError: (error: unknown) => {
        console.error('SVG generation failed:', error);
        alert('Failed to generate SVG. Please try again.');
      },
    }
  );

  const handleContentTypeSelect = (type: 'image_comprehension' | 'comic' | 'math' | 'worksheet') => {
    setSelectedContentType(type);
  
    // Set image count based on content type
    const selectedType = contentTypes.find((ct) => ct.id === type);
    if (selectedType) {
      setImageCount(selectedType.minImages);
    }

    // For image comprehension, set specific defaults
    if (type === 'image_comprehension') {
      setImageCount(1);
      setQuestionTypes(['fill_blank']);
    }
  
    setCurrentStep('configure');
  };

  const handleGenerate = () => {
    if (!selectedContentType || !subject || !gradeLevel || !topic) {
      alert('Please fill in all required fields (Subject, Grade Level, Topic)');
      return;
    }

    if (questionTypes.length === 0) {
      alert('Please select at least one question type');
      return;
    }

    // Get image aspect ratio from selected format
    const selectedFormat = imageFormatOptions.find(f => f.value === imageFormat);
    const imageAspectRatio = selectedFormat ? selectedFormat.aspectRatio : { width: 16, height: 9 };

    const request: GenerateSVGRequest = {
      content_type: selectedContentType,
      subject,
      topic,
      grade_level: gradeLevel,
      layout_style: layoutStyle,
      num_questions: numQuestions, // Will be passed to AI
      question_types: questionTypes, // Will be passed to AI
      image_format: imageFormat, // Changed from aspect_ratio
      image_aspect_ratio: imageAspectRatio, // New field for AI
      image_count: imageCount,
      custom_instructions: customInstructions || undefined,
    };

    generateMutation.mutate(request);
  };

  const handleEditText = () => {
    setCurrentStep('edit');
  };

  const handleExport = () => {
    setCurrentStep('export');
  };

  const handleBackToSelect = () => {
    setCurrentStep('select');
    setSelectedContentType('');
    setGeneratedSVG(null);
    setProcessedSVG('');
  };

  // Updated options for K-6 education
  const gradeOptions = [
    'Kindergarten', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5', 'Grade 6'
  ];

  const subjectOptions = [
    'Mathematics', 'Language Arts', 'Science', 'Social Studies'
  ];

  const layoutOptions = [
    { value: 'layout1', label: 'Fill-in-the-Blanks (Scene-based questions)' },
    { value: 'layout2', label: 'Reading Comprehension (Paragraph + Q&A)' },
    { value: 'layout3', label: 'Multiple Choice Analysis' }
  ];

  const imageFormatOptions = [
    { value: 'landscape', label: 'Landscape Image (16:9)', aspectRatio: { width: 16, height: 9 } },
    { value: 'square', label: 'Square Image (1:1)', aspectRatio: { width: 1, height: 1 } },
    { value: 'portrait', label: 'Portrait Image (3:4)', aspectRatio: { width: 3, height: 4 } },
    { value: 'wide', label: 'Wide Image (3:2)', aspectRatio: { width: 3, height: 2 } }
  ];

  const questionTypeOptions = [
    { value: 'fill_blank', label: 'Fill in the Blanks' },
    { value: 'short_answer', label: 'Short Answer' },
    { value: 'multiple_choice', label: 'Multiple Choice' },
    { value: 'true_false', label: 'True/False' },
    { value: 'matching', label: 'Matching' }
  ];

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
      <div className="flex items-center gap-3 mb-6">
        <FileText className="h-5 w-5 text-blue-600" />
        <h2 className="text-xl font-semibold text-gray-900">K-6 Worksheet Generator</h2>
        {currentStep !== 'select' && (
          <Button
            variant="outline"
            size="sm"
            onClick={handleBackToSelect}
            className="ml-auto"
          >
            Start Over
          </Button>
        )}
      </div>

      {/* Step Indicator */}
      <div className="mb-8">
        <div className="flex items-center justify-between text-sm">
          <div className={`flex items-center ${currentStep === 'select' ? 'text-blue-600 font-medium' : currentStep === 'configure' || currentStep === 'preview' || currentStep === 'edit' || currentStep === 'export' ? 'text-green-600' : 'text-gray-400'}`}>
            <div className={`w-6 h-6 rounded-full flex items-center justify-center mr-2 ${currentStep === 'select' ? 'bg-blue-100 border-2 border-blue-600' : currentStep === 'configure' || currentStep === 'preview' || currentStep === 'edit' || currentStep === 'export' ? 'bg-green-100 border-2 border-green-600' : 'bg-gray-100 border-2 border-gray-300'}`}>
              1
            </div>
            Select Type
          </div>
          <div className={`flex items-center ${currentStep === 'configure' ? 'text-blue-600 font-medium' : currentStep === 'preview' || currentStep === 'edit' || currentStep === 'export' ? 'text-green-600' : 'text-gray-400'}`}>
            <div className={`w-6 h-6 rounded-full flex items-center justify-center mr-2 ${currentStep === 'configure' ? 'bg-blue-100 border-2 border-blue-600' : currentStep === 'preview' || currentStep === 'edit' || currentStep === 'export' ? 'bg-green-100 border-2 border-green-600' : 'bg-gray-100 border-2 border-gray-300'}`}>
              2
            </div>
            Configure
          </div>
          <div className={`flex items-center ${currentStep === 'preview' ? 'text-blue-600 font-medium' : currentStep === 'edit' || currentStep === 'export' ? 'text-green-600' : 'text-gray-400'}`}>
            <div className={`w-6 h-6 rounded-full flex items-center justify-center mr-2 ${currentStep === 'preview' ? 'bg-blue-100 border-2 border-blue-600' : currentStep === 'edit' || currentStep === 'export' ? 'bg-green-100 border-2 border-green-600' : 'bg-gray-100 border-2 border-gray-300'}`}>
              3
            </div>
            Preview
          </div>
          <div className={`flex items-center ${currentStep === 'edit' ? 'text-blue-600 font-medium' : currentStep === 'export' ? 'text-green-600' : 'text-gray-400'}`}>
            <div className={`w-6 h-6 rounded-full flex items-center justify-center mr-2 ${currentStep === 'edit' ? 'bg-blue-100 border-2 border-blue-600' : currentStep === 'export' ? 'bg-green-100 border-2 border-green-600' : 'bg-gray-100 border-2 border-gray-300'}`}>
              4
            </div>
            Edit Text
          </div>
          <div className={`flex items-center ${currentStep === 'export' ? 'text-blue-600 font-medium' : 'text-gray-400'}`}>
            <div className={`w-6 h-6 rounded-full flex items-center justify-center mr-2 ${currentStep === 'export' ? 'bg-blue-100 border-2 border-blue-600' : 'bg-gray-100 border-2 border-gray-300'}`}>
              5
            </div>
            Export
          </div>
        </div>
      </div>

      {/* Step 1: Content Type Selection */}
      {currentStep === 'select' && (
        <ContentTypeSelector
          selectedType={selectedContentType}
          onTypeSelect={handleContentTypeSelect}
        />
      )}

      {/* Step 2: Configuration */}
      {currentStep === 'configure' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">Configure Your K-6 Worksheet</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="subject" className="block text-sm font-medium text-gray-700 mb-2">
                  Subject *
                </label>
                <select
                  id="subject"
                  value={subject}
                  onChange={(e) => setSubject(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                >
                  <option value="">Select subject</option>
                  {subjectOptions.map((subj) => (
                    <option key={subj} value={subj}>{subj}</option>
                  ))}
                </select>
              </div>

              <div>
                <label htmlFor="gradeLevel" className="block text-sm font-medium text-gray-700 mb-2">
                  Grade Level *
                </label>
                <select
                  id="gradeLevel"
                  value={gradeLevel}
                  onChange={(e) => setGradeLevel(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                >
                  <option value="">Select grade level</option>
                  {gradeOptions.map((grade) => (
                    <option key={grade} value={grade}>{grade}</option>
                  ))}
                </select>
              </div>

              <div>
                <label htmlFor="topic" className="block text-sm font-medium text-gray-700 mb-2">
                  Topic *
                </label>
                <input
                  type="text"
                  id="topic"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  placeholder="e.g., Addition & Subtraction, Reading Comprehension, Weather"
                  className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>

              <div>
                <label htmlFor="layoutStyle" className="block text-sm font-medium text-gray-700 mb-2">
                  Worksheet Layout Style
                </label>
                <select
                  id="layoutStyle"
                  value={layoutStyle}
                  onChange={(e) => setLayoutStyle(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {layoutOptions.map((layout) => (
                    <option key={layout.value} value={layout.value}>{layout.label}</option>
                  ))}
                </select>
              </div>

              <div className={`grid gap-4 ${selectedContentType === 'image_comprehension' ? 'grid-cols-1' : 'grid-cols-2'}`}>
                <div>
                  <label htmlFor="numQuestions" className="block text-sm font-medium text-gray-700 mb-2">
                    Number of Questions *
                  </label>
                  <input
                    type="number"
                    id="numQuestions"
                    value={numQuestions}
                    onChange={(e) => {
                      const value = e.target.value;
                      if (value === '') {
                        setNumQuestions(5); // Default value when cleared
                      } else {
                        const parsed = parseInt(value);
                        if (!isNaN(parsed) && parsed >= 3 && parsed <= 8) {
                          setNumQuestions(parsed);
                        }
                      }
                    }}
                    min="3"
                    max="8"
                    className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                
                {selectedContentType !== 'image_comprehension' && (
                  <div>
                    <label htmlFor="imageCount" className="block text-sm font-medium text-gray-700 mb-2">
                      Number of Images
                    </label>
                    <input
                      type="number"
                      id="imageCount"
                      value={imageCount}
                      onChange={(e) => {
                        const value = e.target.value;
                        const minImages = contentTypes.find((ct) => ct.id === selectedContentType)?.minImages ?? 1;
                        const maxImages = contentTypes.find((ct) => ct.id === selectedContentType)?.maxImages ?? 10;
                        
                        if (value === '') {
                          setImageCount(minImages); // Default to minimum when cleared
                        } else {
                          const parsed = parseInt(value);
                          if (!isNaN(parsed) && parsed >= minImages && parsed <= maxImages) {
                            setImageCount(parsed);
                          }
                        }
                      }}
                      min={contentTypes.find((ct) => ct.id === selectedContentType)?.minImages ?? 1}
                      max={contentTypes.find((ct) => ct.id === selectedContentType)?.maxImages ?? 10}
                      className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                )}
              </div>

              <div>
                <label htmlFor="questionTypes" className="block text-sm font-medium text-gray-700 mb-2">
                  {selectedContentType === 'image_comprehension' ? 'Question Type *' : 'Question Types * (AI will respect these)'}
                </label>
                {selectedContentType === 'image_comprehension' ? (
                  <select
                    id="questionTypes"
                    value={questionTypes[0] || 'fill_blank'}
                    onChange={(e) => setQuestionTypes([e.target.value])}
                    className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    required
                  >
                    {questionTypeOptions.map((type) => (
                      <option key={type.value} value={type.value}>{type.label}</option>
                    ))}
                  </select>
                ) : (
                  <>
                    <select
                      id="questionTypes"
                      multiple
                      value={questionTypes}
                      onChange={(e) => setQuestionTypes(Array.from(e.target.selectedOptions, option => option.value))}
                      className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent h-24"
                      required
                    >
                      {questionTypeOptions.map((type) => (
                        <option key={type.value} value={type.value}>{type.label}</option>
                      ))}
                    </select>
                    <p className="text-xs text-gray-500 mt-1">Hold Ctrl/Cmd to select multiple types</p>
                  </>
                )}
              </div>

              <div>
                <label htmlFor="imageFormat" className="block text-sm font-medium text-gray-700 mb-2">
                  Image Format (for AI generation)
                </label>
                <select
                  id="imageFormat"
                  value={imageFormat}
                  onChange={(e) => setImageFormat(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {imageFormatOptions.map((format) => (
                    <option key={format.value} value={format.value}>{format.label}</option>
                  ))}
                </select>
                <p className="text-xs text-gray-500 mt-1">This determines the aspect ratio for AI-generated images</p>
              </div>
            </div>

            <div className="mt-6">
              <label htmlFor="customInstructions" className="block text-sm font-medium text-gray-700 mb-2">
                Custom Instructions (Optional)
              </label>
              <textarea
                id="customInstructions"
                value={customInstructions}
                onChange={(e) => setCustomInstructions(e.target.value)}
                placeholder="Add any specific requirements for your K-6 worksheet..."
                rows={4}
                className="w-full px-4 py-3 border border-gray-200 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>

          <div className="flex gap-3">
            <Button
              variant="outline"
              onClick={() => setCurrentStep('select')}
            >
              Back
            </Button>
            <Button
              onClick={handleGenerate}
              disabled={!subject || !gradeLevel || !topic || questionTypes.length === 0 || generateMutation.isLoading}
              className="flex-1"
            >
              {generateMutation.isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating K-6 Worksheet...
                </>
              ) : (
                <>
                  <Sparkles className="mr-2 h-4 w-4" />
                  Generate K-6 Worksheet
                </>
              )}
            </Button>
          </div>
        </div>
      )}

      {/* Step 3: Preview */}
      {currentStep === 'preview' && generatedSVG && (
        <div className="space-y-6">
          <SVGPreview
            svgContent={generatedSVG.svg_content}
            placeholders={generatedSVG.placeholders}
            onEditText={handleEditText}
            onExport={handleExport}
          />
        </div>
      )}

      {/* Step 4: Text Editing */}
      {currentStep === 'edit' && generatedSVG && (
        <div className="space-y-6">
          <SVGTextEditor
            svgContent={generatedSVG.svg_content}
            placeholders={generatedSVG.placeholders}
            onSave={(processedContent: string) => {
              setProcessedSVG(processedContent);
              setCurrentStep('export');
            }}
            onCancel={() => setCurrentStep('preview')}
          />
        </div>
      )}

      {/* Step 5: Export */}
      {currentStep === 'export' && (generatedSVG || processedSVG) && (
        <div className="space-y-6">
          <SVGExportManager
            svgContent={processedSVG || generatedSVG!.svg_content}
            onBack={() => setCurrentStep(processedSVG ? 'edit' : 'preview')}
          />
        </div>
      )}
    </div>
  );
};