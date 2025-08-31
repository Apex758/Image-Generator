import React, { useState } from 'react';
import { Button } from '../ui/Button';
import { ImagePrompt } from '../../lib/api';
import { Check, X, Edit2, Save } from 'lucide-react';

interface PromptApprovalProps {
  prompts: ImagePrompt[];
  onGenerateApproved: (approvedPrompts: ImagePrompt[]) => void;
}

interface PromptWithStatus extends ImagePrompt {
  isApproved: boolean;
  isEditing: boolean;
  editedPrompt: string;
}

export const PromptApproval: React.FC<PromptApprovalProps> = ({ 
  prompts, 
  onGenerateApproved 
}) => {
  const [promptsWithStatus, setPromptsWithStatus] = useState<PromptWithStatus[]>(
    prompts.map(prompt => ({
      ...prompt,
      isApproved: false,
      isEditing: false,
      editedPrompt: prompt.prompt
    }))
  );

  const handleApprove = (index: number) => {
    setPromptsWithStatus(prev => {
      const updated = [...prev];
      updated[index] = {
        ...updated[index],
        isApproved: true
      };
      return updated;
    });
  };

  const handleReject = (index: number) => {
    setPromptsWithStatus(prev => {
      const updated = [...prev];
      updated[index] = {
        ...updated[index],
        isApproved: false
      };
      return updated;
    });
  };

  const handleEdit = (index: number) => {
    setPromptsWithStatus(prev => {
      const updated = [...prev];
      updated[index] = {
        ...updated[index],
        isEditing: true
      };
      return updated;
    });
  };

  const handleSaveEdit = (index: number) => {
    setPromptsWithStatus(prev => {
      const updated = [...prev];
      updated[index] = {
        ...updated[index],
        isEditing: false,
        prompt: updated[index].editedPrompt
      };
      return updated;
    });
  };

  const handlePromptChange = (index: number, value: string) => {
    setPromptsWithStatus(prev => {
      const updated = [...prev];
      updated[index] = {
        ...updated[index],
        editedPrompt: value
      };
      return updated;
    });
  };

  const handleGenerateApproved = () => {
    const approvedPrompts = promptsWithStatus
      .filter(prompt => prompt.isApproved)
      .map(({ prompt, explanation }) => ({ prompt, explanation }));
    
    onGenerateApproved(approvedPrompts);
  };

  const approvedCount = promptsWithStatus.filter(p => p.isApproved).length;

  return (
    <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Review Generated Image Prompts</h2>
      <p className="text-gray-600 mb-6">
        Review each prompt, edit if needed, and approve or reject. Only approved prompts will be used for image generation.
      </p>
      
      <div className="space-y-4">
        {promptsWithStatus.map((prompt, index) => (
          <div 
            key={index} 
            className={`p-4 rounded-lg border ${
              prompt.isApproved 
                ? 'border-green-200 bg-green-50' 
                : 'border-gray-200 bg-gray-50'
            }`}
          >
            <div className="flex justify-between items-start mb-2">
              <div className="flex-1">
                {prompt.isEditing ? (
                  <div className="mb-2">
                    <label htmlFor={`prompt-${index}`} className="block text-sm font-medium text-gray-700 mb-1">
                      Edit Prompt
                    </label>
                    <textarea
                      id={`prompt-${index}`}
                      className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                      value={prompt.editedPrompt}
                      onChange={(e) => handlePromptChange(index, e.target.value)}
                      rows={3}
                    />
                  </div>
                ) : (
                  <p className="font-medium text-gray-800">{prompt.prompt}</p>
                )}
                <p className="text-sm text-gray-600 mt-1">{prompt.explanation}</p>
              </div>
              
              <div className="flex space-x-2 ml-4">
                {prompt.isEditing ? (
                  <Button 
                    variant="primary" 
                    size="sm" 
                    onClick={() => handleSaveEdit(index)}
                    className="flex items-center"
                  >
                    <Save className="h-4 w-4 mr-1" />
                    Save
                  </Button>
                ) : (
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={() => handleEdit(index)}
                    className="flex items-center"
                  >
                    <Edit2 className="h-4 w-4 mr-1" />
                    Edit
                  </Button>
                )}
                
                <Button 
                  variant={prompt.isApproved ? "primary" : "outline"} 
                  size="sm" 
                  onClick={() => handleApprove(index)}
                  className="flex items-center"
                >
                  <Check className="h-4 w-4 mr-1" />
                  Approve
                </Button>
                
                <Button 
                  variant={!prompt.isApproved ? "secondary" : "outline"} 
                  size="sm" 
                  onClick={() => handleReject(index)}
                  className="flex items-center"
                >
                  <X className="h-4 w-4 mr-1" />
                  Reject
                </Button>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="mt-6 flex justify-between items-center">
        <p className="text-sm text-gray-600">
          {approvedCount} of {promptsWithStatus.length} prompts approved
        </p>
        <Button 
          variant="primary" 
          size="md" 
          onClick={handleGenerateApproved}
          disabled={approvedCount === 0}
          className="flex items-center"
        >
          Generate All Approved Images ({approvedCount})
        </Button>
      </div>
    </div>
  );
};