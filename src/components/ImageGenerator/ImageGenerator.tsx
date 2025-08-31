import React, { useState } from 'react';
import { useMutation, useQueryClient } from 'react-query';
import { Button } from '../ui/Button';
import { Loader2, Sparkles, Settings } from 'lucide-react';
import { api, GenerateImageRequest } from '../../lib/api';

export const ImageGenerator: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [settings, setSettings] = useState({
    width: 1024,
    height: 1024,
    guidance_scale: 3.5,
    num_inference_steps: 50,
    seed: undefined as number | undefined,
  });

  const queryClient = useQueryClient();

  const generateMutation = useMutation(
    (request: GenerateImageRequest) => api.generate(request),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('images');
        setPrompt('');
      },
      onError: (error: any) => {
        console.error('Generation failed:', error);
        alert('Failed to generate image. Please try again.');
      },
    }
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    const request: GenerateImageRequest = {
      prompt: prompt.trim(),
      ...settings,
      seed: settings.seed || undefined,
    };

    generateMutation.mutate(request);
  };

  const presetPrompts = [
    "A serene mountain landscape at sunset",
    "A cozy coffee shop in autumn",
    "A futuristic city with flying cars",
    "A magical forest with glowing mushrooms",
    "A cat wearing a wizard hat",
  ];

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center gap-2 mb-6">
        <Sparkles className="h-5 w-5 text-purple-600" />
        <h2 className="text-xl font-semibold text-gray-900">Generate Image</h2>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="prompt" className="block text-sm font-medium text-gray-700 mb-2">
            Describe your image
          </label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="A homeless cat holding a cardboard sign that says 'Hi Mom!'"
            className="w-full h-24 px-3 py-2 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            required
          />
        </div>

        {/* Preset prompts */}
        <div>
          <p className="text-sm text-gray-600 mb-2">Try these prompts:</p>
          <div className="flex flex-wrap gap-2">
            {presetPrompts.map((preset, index) => (
              <button
                key={index}
                type="button"
                onClick={() => setPrompt(preset)}
                className="text-xs px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-full transition-colors"
              >
                {preset}
              </button>
            ))}
          </div>
        </div>

        {/* Advanced settings */}
        <div>
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
          >
            <Settings className="h-4 w-4" />
            Advanced Settings
            <span className={`transform transition-transform ${showAdvanced ? 'rotate-180' : ''}`}>
              ▼
            </span>
          </button>

          {showAdvanced && (
            <div className="mt-4 p-4 bg-gray-50 rounded-lg space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Width
                  </label>
                  <input
                    type="number"
                    value={settings.width}
                    onChange={(e) => setSettings(prev => ({ ...prev, width: parseInt(e.target.value) }))}
                    min="256"
                    max="2048"
                    step="64"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Height
                  </label>
                  <input
                    type="number"
                    value={settings.height}
                    onChange={(e) => setSettings(prev => ({ ...prev, height: parseInt(e.target.value) }))}
                    min="256"
                    max="2048"
                    step="64"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Guidance Scale: {settings.guidance_scale}
                </label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="0.5"
                  value={settings.guidance_scale}
                  onChange={(e) => setSettings(prev => ({ ...prev, guidance_scale: parseFloat(e.target.value) }))}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Inference Steps: {settings.num_inference_steps}
                </label>
                <input
                  type="range"
                  min="10"
                  max="100"
                  step="5"
                  value={settings.num_inference_steps}
                  onChange={(e) => setSettings(prev => ({ ...prev, num_inference_steps: parseInt(e.target.value) }))}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Seed (optional)
                </label>
                <input
                  type="number"
                  value={settings.seed || ''}
                  onChange={(e) => setSettings(prev => ({ ...prev, seed: e.target.value ? parseInt(e.target.value) : undefined }))}
                  placeholder="Random if empty"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
              </div>
            </div>
          )}
        </div>

        <Button
          type="submit"
          disabled={!prompt.trim() || generateMutation.isLoading}
          className="w-full"
          size="lg"
        >
          {generateMutation.isLoading ? (
            <>
              <Loader2 className="mr-2 h-5 w-5 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Sparkles className="mr-2 h-5 w-5" />
              Generate Image
            </>
          )}
        </Button>
      </form>

      {generateMutation.isError && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-800 text-sm">
            Failed to generate image. Please try again.
          </p>
        </div>
      )}
    </div>
  );
};