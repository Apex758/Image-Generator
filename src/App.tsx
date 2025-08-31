import React, { useEffect, useState } from 'react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Container } from './components/layout/Container';
import { ImageGenerator } from './components/ImageGenerator/ImageGenerator';
import { ImageGallery } from './components/ImageGallery/ImageGallery';
import { Sparkles, Cloud, Cpu } from 'lucide-react';
import { api } from './lib/api';

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

function AppContent() {
  const [config, setConfig] = useState<ConfigData | null>(null);

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await api.getConfig();
        setConfig(response.data);
      } catch (error) {
        console.error('Failed to fetch config:', error);
      }
    };
    fetchConfig();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50">
      <Container className="py-8 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-r from-purple-600 to-blue-600 rounded-xl">
              <Sparkles className="h-8 w-8 text-white" />
            </div>
          </div>
          <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl mb-4">
            FLUX Image Generator
          </h1>
          <p className="text-lg leading-8 text-gray-600 max-w-2xl mx-auto mb-4">
            Create stunning AI-generated images using FLUX.1-dev model. 
            Describe your vision and watch it come to life.
          </p>
          
          {/* Generation Method Indicator */}
          {config && (
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-white rounded-lg border border-gray-200 shadow-sm">
              {config.use_hf_api ? (
                <>
                  <Cloud className="h-4 w-4 text-blue-600" />
                  <span className="text-sm text-gray-700">
                    Using Hugging Face API
                    {!config.hf_api_configured && (
                      <span className="text-red-600 ml-1">(Not Configured)</span>
                    )}
                  </span>
                </>
              ) : (
                <>
                  <Cpu className="h-4 w-4 text-green-600" />
                  <span className="text-sm text-gray-700">Using Local Model</span>
                </>
              )}
            </div>
          )}
        </div>

        {/* Configuration Warning */}
        {config && config.use_hf_api && !config.hf_api_configured && (
          <div className="mb-8 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-start">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-yellow-800">
                  Hugging Face API Token Required
                </h3>
                <div className="mt-2 text-sm text-yellow-700">
                  <p>
                    To use the Hugging Face API, you need to set your API token. 
                    Get one from <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener noreferrer" className="underline">Hugging Face Settings</a> and set the <code className="bg-yellow-100 px-1 rounded">HF_API_TOKEN</code> environment variable.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Main Content */}
        <div className="space-y-8">
          {/* Image Generator */}
          <ImageGenerator />
          
          {/* Image Gallery */}
          <ImageGallery />
        </div>

        {/* Footer */}
        <div className="mt-16 text-center text-sm text-gray-500">
          <p>Powered by FLUX.1-dev • Built with React & FastAPI</p>
          {config && (
            <p className="mt-1">
              Currently using: {config.generation_method}
            </p>
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