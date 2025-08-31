import React, { useState, useEffect } from 'react';
import { useMutation, useQueryClient } from 'react-query';
import { Button } from '../ui/Button';
import { Loader2, CheckCircle, AlertCircle, Image as ImageIcon } from 'lucide-react';
import { api, ImagePrompt, ImageData } from '../../lib/api';

interface SequentialGeneratorProps {
  prompts: ImagePrompt[];
  onComplete: () => void;
  onCancel: () => void;
}

export const SequentialGenerator: React.FC<SequentialGeneratorProps> = ({
  prompts,
  onComplete,
  onCancel
}) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [generatedImages, setGeneratedImages] = useState<ImageData[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const queryClient = useQueryClient();

  // Generate a single image
  const generateMutation = useMutation(
    (request: { prompt: string }) => api.generate({
      prompt: request.prompt,
      width: 1024,
      height: 1024,
      guidance_scale: 3.5,
      num_inference_steps: 50
    }),
    {
      onSuccess: (data) => {
        // Add the new image to our list
        setGeneratedImages(prev => [...prev, data]);
        
        // Move to the next prompt
        setCurrentIndex(prev => prev + 1);
        
        // Refresh the image gallery
        queryClient.invalidateQueries('images');
      },
      onError: (error: any) => {
        console.error('Generation failed:', error);
        setError('Failed to generate image. Please try again.');
        setIsGenerating(false);
      },
    }
  );

  // Start or continue the generation process
  useEffect(() => {
    if (isGenerating && currentIndex < prompts.length) {
      // Generate the current image
      generateMutation.mutate({ prompt: prompts[currentIndex].prompt });
    } else if (isGenerating && currentIndex >= prompts.length) {
      // All images have been generated
      setIsGenerating(false);
      onComplete();
    }
  }, [isGenerating, currentIndex, prompts, generateMutation, onComplete]);

  // Start the generation process
  const handleStart = () => {
    setIsGenerating(true);
    setError(null);
  };

  // Calculate progress percentage
  const progressPercentage = prompts.length > 0 
    ? Math.min(100, (currentIndex / prompts.length) * 100) 
    : 0;

  return (
    <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Sequential Image Generation</h2>
      
      {/* Progress bar */}
      <div className="w-full bg-gray-200 rounded-full h-2.5 mb-4">
        <div 
          className="bg-blue-600 h-2.5 rounded-full transition-all duration-500 ease-in-out" 
          style={{ width: `${progressPercentage}%` }}
        ></div>
      </div>
      
      {/* Progress status */}
      <div className="mb-6 text-center">
        <p className="text-gray-700">
          {isGenerating 
            ? `Generating image ${currentIndex + 1} of ${prompts.length}` 
            : currentIndex === prompts.length 
              ? 'All images generated successfully!' 
              : `Ready to generate ${prompts.length} images`
          }
        </p>
      </div>
      
      {/* Current prompt being processed */}
      {currentIndex < prompts.length && (
        <div className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-100">
          <h3 className="text-sm font-medium text-blue-800 mb-1">Current Prompt:</h3>
          <p className="text-blue-700">{prompts[currentIndex]?.prompt || 'No prompt selected'}</p>
          {prompts[currentIndex]?.explanation && (
            <p className="text-sm text-blue-600 mt-2">{prompts[currentIndex].explanation}</p>
          )}
        </div>
      )}
      
      {/* Latest generated image */}
      {generatedImages.length > 0 && (
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 mb-2">Latest Generated Image:</h3>
          <div className="border border-gray-200 rounded-lg overflow-hidden">
            <img 
              src={generatedImages[generatedImages.length - 1].url} 
              alt={generatedImages[generatedImages.length - 1].prompt}
              className="w-full h-auto"
            />
          </div>
        </div>
      )}
      
      {/* Error message */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 rounded-lg border border-red-100 flex items-start">
          <AlertCircle className="h-5 w-5 text-red-500 mr-2 flex-shrink-0 mt-0.5" />
          <p className="text-red-700">{error}</p>
        </div>
      )}
      
      {/* Action buttons */}
      <div className="flex justify-between">
        <Button 
          variant="outline" 
          onClick={onCancel}
          disabled={isGenerating}
        >
          {currentIndex === prompts.length ? 'Close' : 'Cancel'}
        </Button>
        
        {currentIndex < prompts.length && (
          <Button 
            variant="primary" 
            onClick={handleStart}
            disabled={isGenerating || prompts.length === 0}
            className="ml-auto"
          >
            {isGenerating ? (
              <>
                <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <ImageIcon className="mr-2 h-5 w-5" />
                {currentIndex === 0 ? 'Start Generation' : 'Continue Generation'}
              </>
            )}
          </Button>
        )}
        
        {currentIndex === prompts.length && (
          <Button 
            variant="primary" 
            onClick={onComplete}
            className="ml-auto"
          >
            <CheckCircle className="mr-2 h-5 w-5" />
            View All Images
          </Button>
        )}
      </div>
    </div>
  );
};