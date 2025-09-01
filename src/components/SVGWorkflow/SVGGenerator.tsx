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
  const [aspectRatio, setAspectRatio] = useState('16:9');
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
  
    setCurrentStep('configure');
  };

  const handleGenerate = () => {
    if (!selectedContentType || !subject || !gradeLevel) {
      alert('Please fill in all required fields');
      return;
    }

    const request: GenerateSVGRequest = {
      content_type: selectedContentType,
      subject,
      topic,
      grade_level: gradeLevel,
      layout_style: layoutStyle,
      num_questions: numQuestions,
      question_types: questionTypes,
      aspect_ratio: aspectRatio,
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

  const gradeOptions = [
    'K-2', '3-5', '6-8', '9-12', 'College', 'Adult Education'
  ];

  const aspectRatioOptions = [
    { value: '16:9', label: 'Widescreen (16:9)' },
    { value: '4:3', label: 'Standard (4:3)' },
    { value: '1:1', label: 'Square (1:1)' },
    { value: '3:4', label: 'Portrait (3:4)' },
  ];

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
      <div className="flex items-center gap-3 mb-6">
        <FileText className="h-5 w-5 text-blue-600" />
        <h2 className="text-xl font-semibold text-gray-900">SVG Worksheet Generator</h2>
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
            <h3 className="text-lg font-medium text-gray-900 mb-4">Configure Your Worksheet</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="subject" className="block text-sm font-medium text-gray-700 mb-2">
                  Subject *
                </label>
                <input
                  type="text"
                  id="subject"
                  value={subject}
                  onChange={(e) => setSubject(e.target.value)}
                  placeholder="e.g., Mathematics, Science, English"
                  className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
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
    placeholder="e.g., Prepositions, Reading Comprehension, Animals"
    className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
    required
  />
</div>

<div>
  <label htmlFor="layoutStyle" className="block text-sm font-medium text-gray-700 mb-2">
    Worksheet Layout
  </label>
  <select
    id="layoutStyle"
    value={layoutStyle}
    onChange={(e) => setLayoutStyle(e.target.value)}
    className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
  >
    <option value="layout1">Fill-in-the-Blanks (Scene-based)</option>
    <option value="layout2">Reading Comprehension (Paragraph + Q&A)</option>
    <option value="layout3">Multiple Choice Analysis</option>
  </select>
</div>

<div className="grid grid-cols-2 gap-4">
  <div>
    <label htmlFor="numQuestions" className="block text-sm font-medium text-gray-700 mb-2">
      Number of Questions
    </label>
    <input
      type="number"
      id="numQuestions"
      value={numQuestions}
      onChange={(e) => setNumQuestions(parseInt(e.target.value))}
      min="3"
      max="10"
      className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
    />
  </div>
  
  <div>
    <label htmlFor="questionTypes" className="block text-sm font-medium text-gray-700 mb-2">
      Question Types
    </label>
    <select
      id="questionTypes"
      multiple
      value={questionTypes}
      onChange={(e) => setQuestionTypes(Array.from(e.target.selectedOptions, option => option.value))}
      className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
    >
      <option value="fill_blank">Fill in the Blanks</option>
      <option value="short_answer">Short Answer</option>
      <option value="multiple_choice">Multiple Choice</option>
      <option value="true_false">True/False</option>
    </select>
  </div>
</div>

              <div>
                <label htmlFor="aspectRatio" className="block text-sm font-medium text-gray-700 mb-2">
                  Page Format
                </label>
                <select
                  id="aspectRatio"
                  value={aspectRatio}
                  onChange={(e) => setAspectRatio(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {aspectRatioOptions.map((ratio) => (
                    <option key={ratio.value} value={ratio.value}>{ratio.label}</option>
                  ))}
                </select>
              </div>

              <div>
                <label htmlFor="imageCount" className="block text-sm font-medium text-gray-700 mb-2">
                  Number of Images
                </label>
                <input
                  type="number"
                  id="imageCount"
                  value={imageCount}
                  onChange={(e) => setImageCount(parseInt(e.target.value))}
                  min={contentTypes.find((ct) => ct.id === selectedContentType)?.minImages ?? 1}
                  max={contentTypes.find((ct) => ct.id === selectedContentType)?.maxImages ?? 10}
                  className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
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
                placeholder="Add any specific requirements or instructions for your worksheet..."
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
              disabled={!subject || !gradeLevel || generateMutation.isLoading}
              className="flex-1"
            >
              {generateMutation.isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating Worksheet...
                </>
              ) : (
                <>
                  <Sparkles className="mr-2 h-4 w-4" />
                  Generate Worksheet
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