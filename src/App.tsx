import React, { useEffect, useState } from 'react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Container } from './components/layout/Container';
import { ImageGenerator } from './components/ImageGenerator/ImageGenerator';
import { ImageGallery } from './components/ImageGallery/ImageGallery';
import { Sparkles } from 'lucide-react';
import { api, ImagePrompt } from './lib/api';
import { LessonPlanButton } from './components/LessonPlanModal/LessonPlanButton';
import { LessonPlan } from './components/LessonPlanModal/mockData';
import { PromptApproval } from './components/PromptApproval/PromptApproval';
import { SequentialGenerator } from './components/SequentialGenerator/SequentialGenerator';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

interface ConfigData {
  use_hf_api: boolean;
  hf_api_configured: boolean;
  generation_method: string;
}

interface LessonPlanState {
  selectedLessonPlan: LessonPlan | null;
  imagePrompts: ImagePrompt[];
  isAnalyzing: boolean;
  error: string | null;
  showPromptApproval: boolean;
  showSequentialGenerator: boolean;
  approvedPrompts: ImagePrompt[];
}

function AppContent() {
  const [config, setConfig] = useState<ConfigData | null>(null);
  const [lessonPlanState, setLessonPlanState] = useState<LessonPlanState>({
    selectedLessonPlan: null,
    imagePrompts: [],
    isAnalyzing: false,
    error: null,
    showPromptApproval: false,
    showSequentialGenerator: false,
    approvedPrompts: []
  });

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const configData = await api.getConfig();
        setConfig(configData);
      } catch (error) {
        console.error('Failed to fetch config:', error);
      }
    };
    fetchConfig();
  }, []);

  const handleLessonPlanSelect = async (lessonPlan: LessonPlan) => {
    try {
      setLessonPlanState({
        selectedLessonPlan: lessonPlan,
        imagePrompts: [],
        isAnalyzing: true,
        error: null,
        showPromptApproval: false,
        showSequentialGenerator: false,
        approvedPrompts: []
      });

      // Call the backend to analyze the lesson plan
      const result = await api.analyzeLessonPlan(lessonPlan.content);
      
      setLessonPlanState(prev => ({
        ...prev,
        imagePrompts: result.image_prompts,
        isAnalyzing: false,
        showPromptApproval: true,
        showSequentialGenerator: false
      }));

      console.log('Lesson plan analyzed successfully:', result);
    } catch (error) {
      console.error('Failed to analyze lesson plan:', error);
      setLessonPlanState(prev => ({
        ...prev,
        isAnalyzing: false,
        error: 'Failed to analyze lesson plan. Please try again.',
        showSequentialGenerator: false
      }));
    }
  };

  const handleGenerateApprovedImages = (approvedPrompts: ImagePrompt[]) => {
    console.log('Generating approved images:', approvedPrompts);
    
    // Hide the prompt approval and show the sequential generator
    setLessonPlanState(prev => ({
      ...prev,
      showPromptApproval: false,
      showSequentialGenerator: true,
      approvedPrompts: approvedPrompts
    }));
  };
  
  const handleGenerationComplete = () => {
    // Hide the sequential generator when complete
    setLessonPlanState(prev => ({
      ...prev,
      showSequentialGenerator: false
    }));
  };
  
  const handleGenerationCancel = () => {
    // Go back to prompt approval
    setLessonPlanState(prev => ({
      ...prev,
      showPromptApproval: true,
      showSequentialGenerator: false
    }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-blue-50">
      <Container className="py-10 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-5">
            <div className="p-4 bg-gradient-to-r from-blue-600 to-blue-500 rounded-2xl shadow-md">
              <Sparkles className="h-8 w-8 text-white" />
            </div>
          </div>
          <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl mb-4">
            Educator's Content Studio
          </h1>
          <p className="text-lg leading-8 text-gray-600 max-w-2xl mx-auto mb-5">
            Create professional teaching resources in seconds.
            Generate images, interactive worksheets, and educational content with AI assistance.
          </p>
          
          <div className="flex justify-center gap-4">
            <LessonPlanButton
              onSelectLessonPlan={handleLessonPlanSelect}
              variant="primary"
              size="md"
            />
          </div>
        </div>

        {/* Configuration Warning */}
        {config && config.use_hf_api && !config.hf_api_configured && (
          <div className="mb-8 p-5 bg-yellow-50 border border-yellow-100 rounded-lg shadow-sm">
            <div className="flex items-start">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-yellow-800">
                  API Configuration Required
                </h3>
                <div className="mt-2 text-sm text-yellow-700">
                  <p>
                    To enable cloud-based image generation, you need to set up your API token.
                    Get one from <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener noreferrer" className="underline hover:text-yellow-800 transition-colors">Hugging Face Settings</a> and set the <code className="bg-yellow-100 px-1.5 py-0.5 rounded font-mono text-xs">HF_API_TOKEN</code> environment variable.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Main Content */}
        <div className="space-y-8">
          {/* Selected Lesson Plan Info */}
          {lessonPlanState.selectedLessonPlan && (
            <div className="p-5 bg-blue-50 border border-blue-100 rounded-lg shadow-sm mb-6">
              <h3 className="text-lg font-medium text-blue-800 mb-2">
                Selected Lesson Plan: {lessonPlanState.selectedLessonPlan.title}
              </h3>
              <div className="flex gap-2 mb-3">
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                  {lessonPlanState.selectedLessonPlan.gradeLevel}
                </span>
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                  {lessonPlanState.selectedLessonPlan.subject}
                </span>
              </div>
              
              {lessonPlanState.isAnalyzing && (
                <p className="text-sm text-blue-600">Analyzing lesson plan...</p>
              )}
              
              {lessonPlanState.error && (
                <p className="text-sm text-red-600">{lessonPlanState.error}</p>
              )}
              
              {lessonPlanState.showPromptApproval && lessonPlanState.imagePrompts.length > 0 && (
                <div className="mt-3">
                  <PromptApproval
                    prompts={lessonPlanState.imagePrompts}
                    onGenerateApproved={handleGenerateApprovedImages}
                  />
                </div>
              )}
              
              {lessonPlanState.showSequentialGenerator && lessonPlanState.approvedPrompts.length > 0 && (
                <div className="mt-3">
                  <SequentialGenerator
                    prompts={lessonPlanState.approvedPrompts}
                    onComplete={handleGenerationComplete}
                    onCancel={handleGenerationCancel}
                  />
                </div>
              )}
            </div>
          )}
          
          {/* Image Generator */}
          <ImageGenerator />
          
          {/* Image Gallery */}
          <ImageGallery />
        </div>

        {/* Footer */}
        <div className="mt-16 text-center text-sm text-gray-500">
          {config && (
            <div className="mt-1 space-y-1">
              <p>
                Generation method: <span className="text-blue-600">{config.generation_method}</span>
              </p>
              <p className="text-xs">
                Created for educational purposes. All generated content is for classroom use only.
              </p>
            </div>
          )}
        </div>
      </Container>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  );
}

export default App;