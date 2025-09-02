import React from 'react';
import { BookOpen, Smile, Calculator, FileText } from 'lucide-react';

interface ContentTypeSelectorProps {
  selectedType: 'image_comprehension' | 'comic' | 'math' | 'worksheet' | '';
  onTypeSelect: (type: 'image_comprehension' | 'comic' | 'math' | 'worksheet') => void;
  className?: string;
}

interface ContentTypeOption {
  id: 'image_comprehension' | 'comic' | 'math' | 'worksheet';
  name: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  examples: string[];
  minImages: number;
  maxImages: number;
}

export const contentTypes: ContentTypeOption[] = [
  {
    id: 'image_comprehension',
    name: 'Image Comprehension',
    description: 'Visual reading and comprehension activities for K-6',
    icon: <BookOpen className="h-6 w-6" />,
    color: 'bg-blue-50 border-blue-200 text-blue-700',
    examples: ['Picture analysis', 'Visual storytelling', 'Reading comprehension with images'],
    minImages: 1,
    maxImages: 1,
  },
  {
    id: 'comic',
    name: 'Comic Strip',
    description: 'Sequential storytelling and narrative activities for K-6',
    icon: <Smile className="h-6 w-6" />,
    color: 'bg-purple-50 border-purple-200 text-purple-700',
    examples: ['Story sequences', 'Character development', 'Creative writing prompts'],
    minImages: 2,
    maxImages: 4,
  },
  {
    id: 'math',
    name: 'Math Worksheet',
    description: 'Mathematical problems and visual exercises for K-6',
    icon: <Calculator className="h-6 w-6" />,
    color: 'bg-green-50 border-green-200 text-green-700',
    examples: ['Problem solving', 'Visual math concepts', 'Number sense activities'],
    minImages: 1,
    maxImages: 1,
  },
  {
    id: 'worksheet',
    name: 'Subject Worksheet',
    description: 'Customizable educational worksheets for core K-6 subjects',
    icon: <FileText className="h-6 w-6" />,
    color: 'bg-orange-50 border-orange-200 text-orange-700',
    examples: ['Fill-in-the-blanks', 'Vocabulary practice', 'Subject-specific activities'],
    minImages: 1,
    maxImages: 4,
  }
];

export const ContentTypeSelector: React.FC<ContentTypeSelectorProps> = ({
  selectedType,
  onTypeSelect,
  className = ""
}) => {
  return (
    <div className={`space-y-4 ${className}`}>
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-2">Choose Content Type</h3>
        <p className="text-sm text-gray-600 mb-4">
          Select the type of educational resource you want to create
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {contentTypes.map((type) => (
          <div
            key={type.id}
            className={`
              relative p-4 border-2 rounded-lg cursor-pointer transition-all duration-200
              ${selectedType === type.id 
                ? `${type.color} border-opacity-100 shadow-md` 
                : 'bg-white border-gray-200 hover:border-gray-300 hover:shadow-sm'
              }
            `}
            onClick={() => onTypeSelect(type.id)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onTypeSelect(type.id);
              }
            }}
            aria-label={`Select ${type.name} content type`}
          >
            {/* Selection indicator */}
            {selectedType === type.id && (
              <div className="absolute top-2 right-2">
                <div className="h-3 w-3 bg-current rounded-full"></div>
              </div>
            )}

            <div className="flex items-start gap-3">
              <div className={`
                p-2 rounded-lg 
                ${selectedType === type.id ? 'bg-white/50' : type.color}
              `}>
                {type.icon}
              </div>
              
              <div className="flex-1 min-w-0">
                <h4 className="font-medium text-gray-900 mb-1">{type.name}</h4>
                <p className="text-sm text-gray-600 mb-3">{type.description}</p>
                
                <div className="space-y-1">
                  <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                    Examples:
                  </p>
                  <ul className="text-xs text-gray-600 space-y-0.5">
                    {type.examples.map((example, index) => (
                      <li key={index} className="flex items-center">
                        <span className="w-1 h-1 bg-gray-400 rounded-full mr-2 flex-shrink-0"></span>
                        {example}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {selectedType && (
        <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-sm text-blue-800">
            <span className="font-medium">Selected:</span> {contentTypes.find(t => t.id === selectedType)?.name}
          </p>
        </div>
      )}
    </div>
  );
};