import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { api, ImageData, svgApi, SVGData } from '../../lib/api';
import { Download, Trash2, Copy, Calendar, Settings, Image, FileText, Filter } from 'lucide-react';
import { Button } from '../ui/Button';

interface ImageCardProps {
  image: ImageData;
  onDelete: (id: string) => void;
  onImageClick: (image: ImageData) => void;
}

interface SVGCardProps {
  svg: SVGData;
  onDelete: (id: string) => void;
  onSVGClick: (svg: SVGData) => void;
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

const SVGCard: React.FC<SVGCardProps> = ({ svg, onDelete, onSVGClick }) => {
  const [showDetails, setShowDetails] = useState(false);
  const [content, setContent] = useState<string | null>(null);

  React.useEffect(() => {
    if (svg.url) {
      fetch(svg.url)
        .then((res) => res.text())
        .then(setContent);
    }
  }, [svg.url]);

  const handleDownload = () => {
    if (!content) return;
    const blob = new Blob([content], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `worksheet-${svg.id}.svg`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-100 overflow-hidden group hover:shadow-md transition-all duration-300">
      <div className="relative aspect-square">
        <div
          className="w-full h-full flex items-center justify-center bg-gray-50 cursor-pointer transition-transform duration-500 group-hover:scale-105 p-4"
          onClick={() => onSVGClick(svg)}
        >
          {content ? (
            <div
              className="w-full h-full flex items-center justify-center"
              style={{ transform: 'scale(0.8)' }}
              dangerouslySetInnerHTML={{ __html: content }}
            />
          ) : (
            <div className="animate-pulse bg-gray-200 w-full h-full" />
          )}
        </div>
        <div className="absolute inset-0 bg-gradient-to-t from-black/30 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => onSVGClick(svg)}
            className="bg-white/90 hover:bg-white shadow-md"
          >
            View Worksheet
          </Button>
        </div>
        <div className="absolute top-2 left-2">
          <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
            <FileText className="h-3 w-3 mr-1" />
            SVG
          </span>
        </div>
        <div className="absolute top-2 right-2 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
          <Button
            variant="secondary"
            size="sm"
            onClick={handleDownload}
            className="bg-white/90 hover:bg-white p-2 shadow-sm"
            title="Download worksheet"
          >
            <Download className="h-4 w-4" />
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => onDelete(svg.id)}
            className="bg-red-50/90 hover:bg-red-100 text-red-600 p-2 shadow-sm"
            title="Remove from collection"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="p-4">
        <p className="text-sm text-gray-900 font-medium line-clamp-2 mb-2">
          {svg.template_id.replace('_', ' ')}
        </p>
        <div className="flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center gap-1">
            <Calendar className="h-3 w-3" />
            <span title="Creation date">{formatDate(svg.created_at)}</span>
          </div>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="flex items-center gap-1 hover:text-blue-600 transition-colors"
          >
            <Settings className="h-3 w-3" />
            <span>Details</span>
          </button>
        </div>

        {showDetails && (
          <div className="mt-3 pt-3 border-t border-gray-100 text-xs text-gray-600 space-y-2">
            <div className="flex justify-between">
              <span>Template:</span>
              <span className="capitalize">{svg.template_id.replace('_', ' ')}</span>
            </div>
            {/* Other details can be added here if they are available in the new model */}
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

interface SVGModalProps {
  svg: SVGData | null;
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

const SVGModal: React.FC<SVGModalProps> = ({ svg, onClose }) => {
  const [content, setContent] = useState<string | null>(null);

  React.useEffect(() => {
    if (svg?.url) {
      fetch(svg.url)
        .then((res) => res.text())
        .then(setContent);
    }
  }, [svg?.url]);

  if (!svg) return null;

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
            <p className="font-medium text-gray-900 mb-1 capitalize">
              {svg.template_id.replace('_', ' ')}
            </p>
            <p className="text-sm text-gray-500">
              Created: {new Date(svg.created_at).toLocaleString()}
            </p>
          </div>
          <Button variant="outline" size="sm" onClick={onClose}>
            Close
          </Button>
        </div>
        <div className="max-h-[70vh] overflow-auto bg-gray-50 p-4">
          {content ? (
            <div
              className="flex items-center justify-center"
              style={{ transform: 'scale(0.9)', transformOrigin: 'center' }}
              dangerouslySetInnerHTML={{ __html: content }}
            />
          ) : (
            <div className="animate-pulse bg-gray-200 w-full h-[50vh]" />
          )}
        </div>
        <div className="p-4 bg-white border-t border-gray-100">
          <p className="text-sm text-gray-600">
            This worksheet was generated using the '{svg.template_id}' template.
          </p>
        </div>
      </div>
    </div>
  );
};

export const ImageGallery: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<ImageData | null>(null);
  const [selectedSVG, setSelectedSVG] = useState<SVGData | null>(null);
  const [activeTab, setActiveTab] = useState<'all' | 'images' | 'worksheets'>('all');
  const queryClient = useQueryClient();

  const { data: images = [], isLoading: imagesLoading, error: imagesError } = useQuery(
    'images',
    api.list,
    {
      refetchInterval: 5000, // Refetch every 5 seconds
    }
  );

  const { data: svgItems = [], isLoading: svgLoading, error: svgError } = useQuery(
    'svg-items',
    svgApi.list,
    {
      refetchInterval: 5000, // Refetch every 5 seconds
    }
  );

  const deleteImageMutation = useMutation(
    (imageId: string) => api.delete(imageId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('images');
      },
      onError: (error: unknown) => {
        console.error('Delete failed:', error);
        alert('Failed to delete image. Please try again.');
      },
    }
  );

  const deleteSVGMutation = useMutation(
    (svgId: string) => svgApi.delete(svgId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('svg-items');
      },
      onError: (error: unknown) => {
        console.error('Delete failed:', error);
        alert('Failed to delete worksheet. Please try again.');
      },
    }
  );

  const handleImageDelete = (imageId: string) => {
    if (confirm('Are you sure you want to delete this image?')) {
      deleteImageMutation.mutate(imageId);
    }
  };

  const handleSVGDelete = (svgId: string) => {
    if (confirm('Are you sure you want to delete this worksheet?')) {
      deleteSVGMutation.mutate(svgId);
    }
  };

  const isLoading = imagesLoading || svgLoading;
  const error = imagesError || svgError;

  // Filter items based on active tab
  const filteredImages = activeTab === 'worksheets' ? [] : images;
  const filteredSVGs = activeTab === 'images' ? [] : svgItems;
  const totalItems = images.length + svgItems.length;

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

  if (totalItems === 0) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-8">
        <div className="text-center">
          <p className="text-gray-600">No teaching resources generated yet. Create your first educational content above!</p>
          <p className="text-sm text-gray-500 mt-2">Try generating images for lesson plans or interactive worksheets for your classroom.</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-900">
            Teaching Resources ({totalItems})
          </h2>
          
          {/* Filter Tabs */}
          <div className="flex items-center bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setActiveTab('all')}
              className={`
                flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all duration-200
                ${activeTab === 'all'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
                }
              `}
            >
              <Filter className="h-4 w-4" />
              All ({totalItems})
            </button>
            <button
              onClick={() => setActiveTab('images')}
              className={`
                flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all duration-200
                ${activeTab === 'images'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
                }
              `}
            >
              <Image className="h-4 w-4" />
              Images ({images.length})
            </button>
            <button
              onClick={() => setActiveTab('worksheets')}
              className={`
                flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all duration-200
                ${activeTab === 'worksheets'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
                }
              `}
            >
              <FileText className="h-4 w-4" />
              Worksheets ({svgItems.length})
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {filteredImages.map((image) => (
            <ImageCard
              key={`image-${image.id}`}
              image={image}
              onDelete={handleImageDelete}
              onImageClick={setSelectedImage}
            />
          ))}
          {filteredSVGs.map((svg) => (
            <SVGCard
              key={`svg-${svg.id}`}
              svg={svg}
              onDelete={handleSVGDelete}
              onSVGClick={setSelectedSVG}
            />
          ))}
        </div>
      </div>

      <ImageModal
        image={selectedImage}
        onClose={() => setSelectedImage(null)}
      />
      
      <SVGModal
        svg={selectedSVG}
        onClose={() => setSelectedSVG(null)}
      />
    </>
  );
};