import React from 'react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Container } from './components/layout/Container';
import { ImageGenerator } from './components/ImageGenerator/ImageGenerator';
import { ImageGallery } from './components/ImageGallery/ImageGallery';
import { Sparkles } from 'lucide-react';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function AppContent() {
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
          <p className="text-lg leading-8 text-gray-600 max-w-2xl mx-auto">
            Create stunning AI-generated images using FLUX.1-dev model. 
            Describe your vision and watch it come to life.
          </p>
        </div>

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