import React, { useState, useEffect, useRef } from 'react';
import { useMutation, useQueryClient } from 'react-query';
import { Button } from '../ui/Button';
import { Loader2, Sparkles, Settings, FileText, Image } from 'lucide-react';
import { api, GenerateImageRequest } from '../../lib/api';
import { SVGGenerator } from '../SVGWorkflow/SVGGenerator';

export const ImageGenerator: React.FC = () => {
  const [mode, setMode] = useState<'image' | 'svg'>('image');
  const [prompt, setPrompt] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  // Aspect ratio presets
  type AspectRatioPreset = {
    name: string;
    ratio: number; // width/height
    label: string;
    shape?: React.CSSProperties; // Optional shape styling
  };
  
  const aspectRatioPresets: AspectRatioPreset[] = [
    {
      name: 'square',
      ratio: 1/1,
      label: 'Square (1:1)',
      shape: { width: '16px', height: '16px' }
    },
    {
      name: 'standard',
      ratio: 4/3,
      label: 'Standard (4:3)',
      shape: { width: '16px', height: '12px' }
    },
    {
      name: 'widescreen',
      ratio: 16/9,
      label: 'Widescreen (16:9)',
      shape: { width: '16px', height: '9px' }
    },
    {
      name: 'portrait',
      ratio: 3/4,
      label: 'Portrait (3:4)',
      shape: { width: '12px', height: '16px' }
    },
    {
      name: 'landscape',
      ratio: 3/2,
      label: 'Landscape (3:2)',
      shape: { width: '16px', height: '10.67px' }
    },
    {
      name: 'presentation',
      ratio: 16/10,
      label: 'Presentation (16:10)',
      shape: { width: '16px', height: '10px' }
    },
    {
      name: 'custom',
      ratio: 0,
      label: 'Custom'
    },
  ];
  
  const [selectedAspectRatio, setSelectedAspectRatio] = useState<string>('square');
  const [isCustomRatio, setIsCustomRatio] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  
  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);
  
  // Close dropdown when advanced settings are closed
  useEffect(() => {
    if (!showAdvanced) {
      setDropdownOpen(false);
    }
  }, [showAdvanced]);
  
  // Default settings values
  const defaultSettings = {
    width: 1024,
    height: 1024,
    guidance_scale: 3.5,
    num_inference_steps: 50,
    seed: undefined as number | undefined,
  };
  
  const [settings, setSettings] = useState({...defaultSettings});
  
  // Function to reset settings to defaults
  const resetToDefaults = () => {
    setSettings({...defaultSettings});
    setSelectedAspectRatio('square');
    setIsCustomRatio(false);
  };

  const queryClient = useQueryClient();

  const generateMutation = useMutation(
    (request: GenerateImageRequest) => api.generate(request),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('images');
        setPrompt('');
      },
      onError: (error: unknown) => {
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
    "A magical library where books float and glow, filled with diverse young students reading, fantasy art style",
    "Dinosaurs roaming through a lush prehistoric jungle with volcanoes in the background, vibrant and detailed",
    "Children of different cultures celebrating around the world map with traditional foods and costumes, colorful art",
    "Solar system with planets, asteroid belts, and a space station, cosmic art with bright colors and detail",
    "Underground ant colony with tunnels, chambers, and busy ants working, detailed cutaway scientific illustration",
    "A time machine workshop filled with gears, steam, and inventors from different eras working together",
    "Ocean ecosystem showing coral reefs, fish, whales, and underwater plants in crystal clear water",
  ];

  // If SVG mode is selected, show the SVG generator
  if (mode === 'svg') {
    return <SVGGenerator />;
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Sparkles className="h-5 w-5 text-blue-600" />
          <h2 className="text-xl font-semibold text-gray-900">Create Teaching Resource</h2>
        </div>
        
        {/* Mode Toggle */}
        <div className="flex items-center bg-gray-100 rounded-lg p-1">
          <button
            type="button"
            onClick={() => setMode('image')}
            className={`
              flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all duration-200
              ${mode === 'image'
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
              }
            `}
          >
            <Image className="h-4 w-4" />
            Images
          </button>
          <button
            type="button"
            onClick={() => setMode('svg')}
            className={`
              flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-all duration-200
              ${mode === 'svg'
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
              }
            `}
          >
            <FileText className="h-4 w-4" />
            Worksheets
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="prompt" className="block text-sm font-medium text-gray-700 mb-2">
            Describe your teaching resource
          </label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Describe the educational image you need for your classroom or lesson plan..."
            className="w-full h-24 px-4 py-3 border border-gray-200 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent shadow-sm"
            required
          />
        </div>

        {/* Preset prompts */}
        <div>
          <p className="text-sm text-gray-600 mb-2">Educational examples:</p>
          <div className="flex flex-wrap gap-2">
            {presetPrompts.map((preset, index) => (
              <button
                key={index}
                type="button"
                onClick={() => setPrompt(preset)}
                className="text-xs px-3 py-1.5 bg-blue-50 text-blue-700 hover:bg-blue-100 rounded-full transition-colors shadow-sm"
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
            className="flex items-center gap-2 text-sm text-gray-600 hover:text-blue-600 transition-colors"
          >
            <Settings className="h-4 w-4" />
            Resource Settings
            <span className={`transform transition-transform ${showAdvanced ? 'rotate-180' : ''}`}>
              ▼
            </span>
          </button>

          {showAdvanced && (
            <div className="mt-4 p-5 bg-gray-50 rounded-lg space-y-4 border border-gray-100">
              {/* Defaults button */}
              <div className="flex justify-end">
                <button
                  type="button"
                  onClick={resetToDefaults}
                  className="text-sm px-3 py-1.5 bg-gray-200 hover:bg-gray-300 rounded-md transition-colors shadow-sm"
                >
                  Reset to Defaults
                </button>
              </div>
              
              {/* Aspect Ratio Selector */}
              <div className="mb-4">
                <label
                  className="block text-sm font-medium text-gray-700 mb-1"
                  title="Select a preset aspect ratio for your teaching resource"
                >
                  Resource Format
                </label>
                <div className="space-y-2">
                  <div className="flex items-center gap-4">
                    {/* Custom dropdown with shapes */}
                    <div className="flex items-center flex-grow relative" ref={dropdownRef}>
                      <div
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent cursor-pointer flex items-center justify-between"
                        onClick={() => setDropdownOpen(!dropdownOpen)}
                      >
                        <div className="flex items-center">
                          {selectedAspectRatio !== 'custom' && (
                            <div
                              className="inline-block mr-2 border border-gray-400 bg-purple-100"
                              style={aspectRatioPresets.find(p => p.name === selectedAspectRatio)?.shape}
                            />
                          )}
                          <span>{aspectRatioPresets.find(p => p.name === selectedAspectRatio)?.label}</span>
                        </div>
                        <span className={`transform transition-transform ${dropdownOpen ? 'rotate-180' : ''}`}>▼</span>
                      </div>
                      
                      {dropdownOpen && (
                        <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-300 rounded-lg shadow-lg z-10">
                          {aspectRatioPresets.map((preset) => (
                            <div
                              key={preset.name}
                              className={`px-3 py-2 cursor-pointer flex items-center hover:bg-gray-100 ${selectedAspectRatio === preset.name ? 'bg-purple-50' : ''}`}
                              onClick={() => {
                                const newRatio = preset.name;
                                setSelectedAspectRatio(newRatio);
                                setIsCustomRatio(newRatio === 'custom');
                                setDropdownOpen(false);
                                
                                // If not custom, adjust dimensions to match the selected ratio
                                if (newRatio !== 'custom') {
                                  // Keep the width and adjust the height based on the ratio
                                  const newHeight = Math.round(settings.width / preset.ratio);
                                  setSettings(prev => ({
                                    ...prev,
                                    height: newHeight
                                  }));
                                }
                              }}
                            >
                              {preset.name !== 'custom' && (
                                <div
                                  className="inline-block mr-2 border border-gray-400 bg-purple-100"
                                  style={preset.shape}
                                />
                              )}
                              <span>{preset.label}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                    
                    {/* Visual indicator on the right */}
                    <div className="flex items-center justify-center min-w-[100px]">
                      {selectedAspectRatio !== 'custom' && (
                        <div
                          className="border-2 border-purple-500 bg-purple-100 rounded-md"
                          style={{
                            width: '80px',
                            height: `${80 / (aspectRatioPresets.find(p => p.name === selectedAspectRatio)?.ratio || 1)}px`,
                            maxHeight: '80px',
                            transition: 'all 0.3s ease'
                          }}
                          title={`Visual representation of ${aspectRatioPresets.find(p => p.name === selectedAspectRatio)?.label}`}
                        />
                      )}
                      {selectedAspectRatio === 'custom' && (
                        <div className="text-sm text-gray-500">
                          {settings.width} × {settings.height}
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <p className="text-xs text-gray-500">
                    Choose the best format for your teaching materials (slides, handouts, posters, etc.)
                  </p>
                </div>
              </div>
              
              {/* Image Size - Most basic and important setting */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label
                    className="block text-sm font-medium text-gray-700 mb-1"
                    title={isCustomRatio
                      ? "The width of your resource in pixels. Standard size is 1024."
                      : "The width of your resource in pixels. Changing this will automatically adjust the height to maintain the selected aspect ratio."}
                  >
                    Resource Width {!isCustomRatio && <span className="text-xs text-blue-600">(linked)</span>}
                  </label>
                  <input
                    type="number"
                    value={settings.width}
                    onChange={(e) => {
                      const newWidth = parseInt(e.target.value);
                      
                      if (isCustomRatio) {
                        // For custom ratio, just update the width
                        setSettings(prev => ({ ...prev, width: newWidth }));
                      } else {
                        // For preset ratios, maintain the aspect ratio
                        const preset = aspectRatioPresets.find(p => p.name === selectedAspectRatio);
                        if (preset) {
                          const newHeight = Math.round(newWidth / preset.ratio);
                          setSettings(prev => ({
                            ...prev,
                            width: newWidth,
                            height: newHeight
                          }));
                        }
                      }
                    }}
                    min="256"
                    max="2048"
                    step="64"
                    className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent shadow-sm"
                    title="Larger values create wider resources but take longer to generate"
                  />
                </div>
                <div>
                  <label
                    className="block text-sm font-medium text-gray-700 mb-1"
                    title={isCustomRatio
                      ? "The height of your resource in pixels. Standard size is 1024."
                      : "The height of your resource in pixels. Changing this will automatically adjust the width to maintain the selected aspect ratio."}
                  >
                    Resource Height {!isCustomRatio && <span className="text-xs text-blue-600">(linked)</span>}
                  </label>
                  <input
                    type="number"
                    value={settings.height}
                    onChange={(e) => {
                      const newHeight = parseInt(e.target.value);
                      
                      if (isCustomRatio) {
                        // For custom ratio, just update the height
                        setSettings(prev => ({ ...prev, height: newHeight }));
                      } else {
                        // For preset ratios, maintain the aspect ratio
                        const preset = aspectRatioPresets.find(p => p.name === selectedAspectRatio);
                        if (preset) {
                          const newWidth = Math.round(newHeight * preset.ratio);
                          setSettings(prev => ({
                            ...prev,
                            width: newWidth,
                            height: newHeight
                          }));
                        }
                      }
                    }}
                    min="256"
                    max="2048"
                    step="64"
                    className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent shadow-sm"
                    title="Larger values create taller resources but take longer to generate"
                  />
                </div>
              </div>

              {/* Seed - Important for reproducibility in education */}
              <div>
                <label
                  className="block text-sm font-medium text-gray-700 mb-1"
                  title="A seed value lets you recreate the same resource again later - useful for creating matching sets"
                >
                  Reproducibility ID (optional)
                </label>
                <input
                  type="number"
                  value={settings.seed || ''}
                  onChange={(e) => setSettings(prev => ({ ...prev, seed: e.target.value ? parseInt(e.target.value) : undefined }))}
                  placeholder="Random if empty"
                  className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent shadow-sm"
                  title="Use the same number to create matching resources. Leave empty for unique results each time."
                />
              </div>

              {/* Guidance Scale - Affects how closely the image follows the prompt */}
              <div>
                <label
                  className="block text-sm font-medium text-gray-700 mb-1"
                  title="Controls how closely the resource follows your description"
                >
                  Description Adherence: {settings.guidance_scale}
                </label>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="0.5"
                  value={settings.guidance_scale}
                  onChange={(e) => setSettings(prev => ({ ...prev, guidance_scale: parseFloat(e.target.value) }))}
                  className="w-full"
                  title="Higher values make the image follow your text more closely. Lower values allow more creativity."
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>More Creative</span>
                  <span>More Literal</span>
                </div>
              </div>

              {/* Inference Steps - More technical, affects quality */}
              <div>
                <label
                  className="block text-sm font-medium text-gray-700 mb-1"
                  title="Controls the quality and detail of the resource"
                >
                  Resource Quality: {settings.num_inference_steps}
                </label>
                <input
                  type="range"
                  min="10"
                  max="100"
                  step="5"
                  value={settings.num_inference_steps}
                  onChange={(e) => setSettings(prev => ({ ...prev, num_inference_steps: parseInt(e.target.value) }))}
                  className="w-full"
                  title="Higher values create more detailed images but take longer to generate"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Faster</span>
                  <span>Higher Quality</span>
                </div>
              </div>
            </div>
          )}
        </div>

        <Button
          type="submit"
          disabled={!prompt.trim() || generateMutation.isLoading}
          className="w-full shadow-sm"
          size="lg"
        >
          {generateMutation.isLoading ? (
            <>
              <Loader2 className="mr-2 h-5 w-5 animate-spin" />
              Creating your resource...
            </>
          ) : (
            <>
              <Sparkles className="mr-2 h-5 w-5" />
              Create Teaching Resource
            </>
          )}
        </Button>
      </form>

      {generateMutation.isError && (
        <div className="mt-4 p-4 bg-red-50 border border-red-100 rounded-lg shadow-sm">
          <p className="text-red-800 text-sm">
            Failed to create teaching resource. Please try again.
          </p>
        </div>
      )}
    </div>
  );
};