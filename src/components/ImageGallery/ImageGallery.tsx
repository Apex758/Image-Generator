import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { api, ImageData } from '../../lib/api';
import { Download, Trash2, Copy, Calendar, Settings } from 'lucide-react';
import { Button } from '../ui/Button';

interface ImageCardProps {
  image: ImageData;
  onDelete: (id: string) => void;
  onImageClick: (image: ImageData) => void;
}

const ImageCard: React.FC<ImageCardProps> = ({ image, onDelete, onImageClick }) => {
  const [showDetails, setShowDetails] = useState(false);

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = image.url;
    link.download = `flux-${image.id}.png`;
    link.click();
  };

  const copyPrompt = () => {
    navigator.clipboard.writeText(image.prompt);
    // You could add a toast notification here
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-100 overflow-hidden group hover:shadow-md transition-all duration-300">
      <div className="relative aspect-square">
        <img
          src={image.url}
          alt={image.prompt}
          className="w-full h-full object-cover cursor-pointer transition-transform duration-500 group-hover:scale-105"
          onClick={() => onImageClick(image)}
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/30 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => onImageClick(image)}
            className="bg-white/90 hover:bg-white shadow-md"
          >
            View Teaching Resource
          </Button>
        </div>
        <div className="absolute top-2 right-2 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
          <Button
            variant="secondary"
            size="sm"
            onClick={handleDownload}
            className="bg-white/90 hover:bg-white p-2 shadow-sm"
            title="Download for classroom use"
          >
            <Download className="h-4 w-4" />
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => onDelete(image.id)}
            className="bg-red-50/90 hover:bg-red-100 text-red-600 p-2 shadow-sm"
            title="Remove from collection"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="p-4">
        <p className="text-sm text-gray-900 font-medium line-clamp-2 mb-2">{image.prompt}</p>
        <div className="flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center gap-1">
            <Calendar className="h-3 w-3" />
            <span title="Creation date">{formatDate(image.created_at)}</span>
          </div>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="flex items-center gap-1 hover:text-blue-600 transition-colors"
          >
            <Settings className="h-3 w-3" />
            <span>Resource Details</span>
          </button>
        </div>

        {showDetails && (
          <div className="mt-3 pt-3 border-t border-gray-100 text-xs text-gray-600 space-y-2">
            <div className="flex justify-between">
              <span>Resource Size:</span>
              <span>{image.width} × {image.height}</span>
            </div>
            {image.guidance_scale && (
              <div className="flex justify-between">
                <span>Prompt Adherence:</span>
                <span>{image.guidance_scale}</span>
              </div>
            )}
            {image.num_inference_steps && (
              <div className="flex justify-between">
                <span>Quality Level:</span>
                <span>{image.num_inference_steps}</span>
              </div>
            )}
            {image.seed && (
              <div className="flex justify-between">
                <span>Consistency ID:</span>
                <span>{image.seed}</span>
              </div>
            )}
            <button
              onClick={copyPrompt}
              className="w-full mt-2 flex items-center justify-center gap-1 py-1.5 px-2 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-md transition-colors"
            >
              <Copy className="h-3 w-3" />
              <span>Copy for Lesson Plan</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

interface ImageModalProps {
  image: ImageData | null;
  onClose: () => void;
}

const ImageModal: React.FC<ImageModalProps> = ({ image, onClose }) => {
  if (!image) return null;

  return (
    <div
      className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-lg max-w-4xl max-h-[90vh] overflow-hidden shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-5 border-b border-gray-100 flex justify-between items-start">
          <div>
            <p className="font-medium text-gray-900 mb-1">{image.prompt}</p>
            <p className="text-sm text-gray-500">
              Resource dimensions: {image.width} × {image.height} • Created: {new Date(image.created_at).toLocaleString()}
            </p>
          </div>
          <Button variant="outline" size="sm" onClick={onClose}>
            Close
          </Button>
        </div>
        <div className="max-h-[70vh] overflow-auto bg-gray-50">
          <img
            src={image.url}
            alt={image.prompt}
            className="w-full h-auto"
          />
        </div>
        <div className="p-4 bg-white border-t border-gray-100">
          <p className="text-sm text-gray-600">
            This teaching resource was generated based on your prompt. You can download it for use in your classroom materials.
          </p>
        </div>
      </div>
    </div>
  );
};

export const ImageGallery: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<ImageData | null>(null);
  const queryClient = useQueryClient();

  const { data: images = [], isLoading, error } = useQuery(
    'images',
    api.list,
    {
      refetchInterval: 5000, // Refetch every 5 seconds
    }
  );

  const deleteMutation = useMutation(
    (imageId: string) => api.delete(imageId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('images');
      },
      onError: (error: any) => {
        console.error('Delete failed:', error);
        alert('Failed to delete image. Please try again.');
      },
    }
  );

  const handleDelete = (imageId: string) => {
    if (confirm('Are you sure you want to delete this image?')) {
      deleteMutation.mutate(imageId);
    }
  };

  if (isLoading) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-8">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading your teaching resources...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-8">
        <div className="text-center">
          <p className="text-red-600">Failed to load teaching resources. Please try again.</p>
        </div>
      </div>
    );
  }

  if (images.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-8">
        <div className="text-center">
          <p className="text-gray-600">No teaching resources generated yet. Create your first educational image above!</p>
          <p className="text-sm text-gray-500 mt-2">Try generating images for lesson plans, classroom decorations, or visual aids.</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-900">
            Teaching Resources ({images.length})
          </h2>
          <p className="text-sm text-gray-500">Resources for your classroom</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {images.map((image) => (
            <ImageCard
              key={image.id}
              image={image}
              onDelete={handleDelete}
              onImageClick={setSelectedImage}
            />
          ))}
        </div>
      </div>

      <ImageModal
        image={selectedImage}
        onClose={() => setSelectedImage(null)}
      />
    </>
  );
};