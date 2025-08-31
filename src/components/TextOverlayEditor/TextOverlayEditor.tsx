import React, { useState, useRef } from 'react';
import { useMutation, useQueryClient } from 'react-query';
import { Button } from '../ui/Button';
import { Save, Type, X, Plus } from 'lucide-react';
import { api, ImageData, TextOverlay, ApplyTextOverlayRequest } from '../../lib/api';

interface TextOverlayEditorProps {
  image: ImageData;
  onClose: () => void;
  onSave?: (newImage: ImageData) => void;
}

export const TextOverlayEditor: React.FC<TextOverlayEditorProps> = ({
  image,
  onClose,
  onSave
}) => {
  const [overlays, setOverlays] = useState<TextOverlay[]>([]);
  const [selectedOverlay, setSelectedOverlay] = useState<string | null>(null);
  const [isAddingText, setIsAddingText] = useState(false);
  const imageRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();

  const saveMutation = useMutation(
    (request: ApplyTextOverlayRequest) => api.applyTextOverlay(request),
    {
      onSuccess: (newImage) => {
        queryClient.invalidateQueries('images');
        onSave?.(newImage);
        onClose();
      },
      onError: (error) => {
        console.error('Failed to save text overlay:', error);
        alert('Failed to save text overlay. Please try again.');
      },
    }
  );

  const handleImageClick = (event: React.MouseEvent<HTMLImageElement>) => {
    if (!isAddingText || !imageRef.current) return;

    const rect = imageRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Calculate relative position (0-1) and convert to actual image coordinates
    const relativeX = x / rect.width;
    const relativeY = y / rect.height;
    const imageX = Math.round(relativeX * image.width);
    const imageY = Math.round(relativeY * image.height);

    const newOverlay: TextOverlay = {
      id: `overlay-${Date.now()}`,
      x: imageX,
      y: imageY,
      width: 200,
      height: 40,
      text: 'Click to edit text',
      fontSize: 16,
      fontFamily: 'arial.ttf',
      color: '#000000',
      alignment: 'left'
    };

    setOverlays(prev => [...prev, newOverlay]);
    setSelectedOverlay(newOverlay.id);
    setIsAddingText(false);
  };

  const updateOverlay = (id: string, updates: Partial<TextOverlay>) => {
    setOverlays(prev => 
      prev.map(overlay => 
        overlay.id === id ? { ...overlay, ...updates } : overlay
      )
    );
  };

  const deleteOverlay = (id: string) => {
    setOverlays(prev => prev.filter(overlay => overlay.id !== id));
    setSelectedOverlay(null);
  };

  const handleSave = () => {
    if (overlays.length === 0) {
      alert('Please add some text before saving.');
      return;
    }

    const request: ApplyTextOverlayRequest = {
      imageId: image.id,
      overlays: overlays
    };

    saveMutation.mutate(request);
  };

  const selectedOverlayData = overlays.find(o => o.id === selectedOverlay);

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg max-w-6xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <div className="flex items-center gap-2">
            <Type className="h-5 w-5 text-blue-600" />
            <h2 className="text-lg font-semibold">Add Text to Image</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-full"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="flex flex-1 overflow-hidden">
          {/* Image Area */}
          <div className="flex-1 p-4 overflow-auto">
            <div className="relative inline-block" ref={containerRef}>
              <img
                ref={imageRef}
                src={image.url}
                alt="Image to edit"
                className={`max-w-full h-auto ${isAddingText ? 'cursor-crosshair' : 'cursor-default'}`}
                onClick={handleImageClick}
              />
              
              {/* Overlay Elements */}
              {overlays.map(overlay => {
                const imageElement = imageRef.current;
                if (!imageElement) return null;

                const rect = imageElement.getBoundingClientRect();
                const containerRect = containerRef.current?.getBoundingClientRect();
                if (!containerRect) return null;

                // Calculate display position
                const scaleX = rect.width / image.width;
                const scaleY = rect.height / image.height;
                const displayX = overlay.x * scaleX;
                const displayY = overlay.y * scaleY;
                const displayWidth = overlay.width * scaleX;
                const displayHeight = overlay.height * scaleY;

                return (
                  <div
                    key={overlay.id}
                    className={`absolute border-2 bg-white bg-opacity-80 p-1 cursor-pointer ${
                      selectedOverlay === overlay.id ? 'border-blue-500' : 'border-gray-300'
                    }`}
                    style={{
                      left: displayX,
                      top: displayY,
                      width: displayWidth,
                      height: displayHeight,
                      fontSize: (overlay.fontSize || 16) * Math.min(scaleX, scaleY),
                      color: overlay.color,
                      textAlign: overlay.alignment as 'left' | 'center' | 'right',
                      overflow: 'hidden'
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedOverlay(overlay.id);
                    }}
                  >
                    {overlay.text}
                  </div>
                );
              })}
            </div>

            {/* Instructions */}
            <div className="mt-4 p-3 bg-blue-50 rounded-lg">
              <p className="text-sm text-blue-800">
                <strong>Instructions:</strong>
                {isAddingText 
                  ? " Click anywhere on the image to add a text box."
                  : " Click 'Add Text' button, then click on the image where you want to place text. Click on existing text boxes to edit them."}
              </p>
            </div>
          </div>

          {/* Controls Panel */}
          <div className="w-80 border-l p-4 overflow-auto">
            <div className="space-y-4">
              {/* Add Text Button */}
              <Button
                onClick={() => setIsAddingText(!isAddingText)}
                variant={isAddingText ? "secondary" : "primary"}
                className="w-full"
              >
                <Plus className="mr-2 h-4 w-4" />
                {isAddingText ? 'Cancel Adding' : 'Add Text'}
              </Button>

              {/* Text Overlay List */}
              <div>
                <h3 className="font-medium mb-2">Text Elements ({overlays.length})</h3>
                {overlays.length === 0 ? (
                  <p className="text-sm text-gray-500">No text added yet</p>
                ) : (
                  <div className="space-y-2">
                    {overlays.map(overlay => (
                      <div
                        key={overlay.id}
                        className={`p-2 border rounded cursor-pointer ${
                          selectedOverlay === overlay.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
                        }`}
                        onClick={() => setSelectedOverlay(overlay.id)}
                      >
                        <div className="flex justify-between items-start">
                          <span className="text-sm font-medium truncate">{overlay.text}</span>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              deleteOverlay(overlay.id);
                            }}
                            className="text-red-500 hover:text-red-700"
                          >
                            <X className="h-3 w-3" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Text Properties */}
              {selectedOverlayData && (
                <div className="border-t pt-4">
                  <h3 className="font-medium mb-3">Edit Text Properties</h3>
                  <div className="space-y-3">
                    <div>
                      <label className="block text-sm font-medium mb-1">Text</label>
                      <textarea
                        value={selectedOverlayData.text}
                        onChange={(e) => updateOverlay(selectedOverlay!, { text: e.target.value })}
                        className="w-full px-3 py-2 border border-gray-200 rounded-lg resize-none"
                        rows={2}
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium mb-1">Font Size</label>
                      <input
                        type="number"
                        value={selectedOverlayData.fontSize}
                        onChange={(e) => updateOverlay(selectedOverlay!, { fontSize: parseInt(e.target.value) })}
                        min="8"
                        max="72"
                        className="w-full px-3 py-2 border border-gray-200 rounded-lg"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium mb-1">Color</label>
                      <input
                        type="color"
                        value={selectedOverlayData.color}
                        onChange={(e) => updateOverlay(selectedOverlay!, { color: e.target.value })}
                        className="w-full h-10 border border-gray-200 rounded-lg"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium mb-1">Alignment</label>
                      <select
                        value={selectedOverlayData.alignment}
                        onChange={(e) => updateOverlay(selectedOverlay!, { alignment: e.target.value as 'left' | 'center' | 'right' })}
                        className="w-full px-3 py-2 border border-gray-200 rounded-lg"
                      >
                        <option value="left">Left</option>
                        <option value="center">Center</option>
                        <option value="right">Right</option>
                      </select>
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className="block text-sm font-medium mb-1">Width</label>
                        <input
                          type="number"
                          value={selectedOverlayData.width}
                          onChange={(e) => updateOverlay(selectedOverlay!, { width: parseInt(e.target.value) })}
                          min="50"
                          max="800"
                          className="w-full px-3 py-2 border border-gray-200 rounded-lg"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium mb-1">Height</label>
                        <input
                          type="number"
                          value={selectedOverlayData.height}
                          onChange={(e) => updateOverlay(selectedOverlay!, { height: parseInt(e.target.value) })}
                          min="20"
                          max="200"
                          className="w-full px-3 py-2 border border-gray-200 rounded-lg"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-4 border-t bg-gray-50">
          <div className="text-sm text-gray-600">
            {overlays.length} text element{overlays.length !== 1 ? 's' : ''} added
          </div>
          <div className="flex gap-2">
            <Button variant="secondary" onClick={onClose}>
              Cancel
            </Button>
            <Button
              onClick={handleSave}
              disabled={overlays.length === 0 || saveMutation.isLoading}
            >
              {saveMutation.isLoading ? (
                <>Saving...</>
              ) : (
                <>
                  <Save className="mr-2 h-4 w-4" />
                  Save with Text
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};