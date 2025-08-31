import React, { useState, useEffect, useRef } from 'react';
import { X, Search, BookOpen, Filter } from 'lucide-react';
import { Button } from '../ui/Button';
import { LessonPlan, lessonPlans, getUniqueGradeLevels, getUniqueSubjects } from './mockData';
import { cn } from '../../lib/utils';

interface LessonPlanModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectLessonPlan: (lessonPlan: LessonPlan) => void;
}

export const LessonPlanModal: React.FC<LessonPlanModalProps> = ({
  isOpen,
  onClose,
  onSelectLessonPlan,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedGradeLevel, setSelectedGradeLevel] = useState<string>('');
  const [selectedSubject, setSelectedSubject] = useState<string>('');
  const [filteredLessonPlans, setFilteredLessonPlans] = useState<LessonPlan[]>(lessonPlans);
  const [selectedLessonPlan, setSelectedLessonPlan] = useState<LessonPlan | null>(null);
  
  const modalRef = useRef<HTMLDivElement>(null);
  const gradeLevels = getUniqueGradeLevels();
  const subjects = getUniqueSubjects();

  // Handle click outside to close modal
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (modalRef.current && !modalRef.current.contains(event.target as Node)) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen, onClose]);

  // Filter lesson plans based on search query and filters
  useEffect(() => {
    let filtered = lessonPlans;
    
    // Apply grade level filter
    if (selectedGradeLevel) {
      filtered = filtered.filter(plan => plan.gradeLevel === selectedGradeLevel);
    }
    
    // Apply subject filter
    if (selectedSubject) {
      filtered = filtered.filter(plan => plan.subject === selectedSubject);
    }
    
    // Apply search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        plan => 
          plan.title.toLowerCase().includes(query) || 
          plan.content.toLowerCase().includes(query)
      );
    }
    
    setFilteredLessonPlans(filtered);
  }, [searchQuery, selectedGradeLevel, selectedSubject]);

  // Reset filters
  const resetFilters = () => {
    setSelectedGradeLevel('');
    setSelectedSubject('');
    setSearchQuery('');
  };

  // Handle lesson plan selection
  const handleSelectLessonPlan = (lessonPlan: LessonPlan) => {
    setSelectedLessonPlan(lessonPlan);
  };

  // Handle confirm selection
  const handleConfirmSelection = () => {
    if (selectedLessonPlan) {
      onSelectLessonPlan(selectedLessonPlan);
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div 
        ref={modalRef}
        className="bg-white rounded-xl shadow-lg w-full max-w-4xl max-h-[90vh] flex flex-col"
      >
        {/* Modal Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <div className="flex items-center gap-2">
            <BookOpen className="h-5 w-5 text-blue-600" />
            <h2 className="text-xl font-semibold text-gray-900">Select Lesson Plan</h2>
          </div>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 transition-colors"
            aria-label="Close modal"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        
        {/* Search and Filters */}
        <div className="p-4 border-b border-gray-200 space-y-4">
          {/* Search */}
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search className="h-5 w-5 text-gray-400" />
            </div>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search lesson plans..."
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          
          {/* Filters */}
          <div className="flex flex-wrap gap-4">
            <div className="flex-1 min-w-[200px]">
              <label htmlFor="gradeLevel" className="block text-sm font-medium text-gray-700 mb-1">
                Grade Level
              </label>
              <select
                id="gradeLevel"
                value={selectedGradeLevel}
                onChange={(e) => setSelectedGradeLevel(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">All Grade Levels</option>
                {gradeLevels.map((grade) => (
                  <option key={grade} value={grade}>
                    {grade}
                  </option>
                ))}
              </select>
            </div>
            
            <div className="flex-1 min-w-[200px]">
              <label htmlFor="subject" className="block text-sm font-medium text-gray-700 mb-1">
                Subject
              </label>
              <select
                id="subject"
                value={selectedSubject}
                onChange={(e) => setSelectedSubject(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">All Subjects</option>
                {subjects.map((subject) => (
                  <option key={subject} value={subject}>
                    {subject}
                  </option>
                ))}
              </select>
            </div>
            
            <div className="flex items-end">
              <Button
                variant="secondary"
                onClick={resetFilters}
                className="flex items-center gap-1"
              >
                <Filter className="h-4 w-4" />
                Reset Filters
              </Button>
            </div>
          </div>
        </div>
        
        {/* Lesson Plans List */}
        <div className="flex-1 overflow-y-auto p-4">
          {filteredLessonPlans.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <p>No lesson plans found matching your criteria.</p>
              <p className="text-sm mt-2">Try adjusting your filters or search query.</p>
            </div>
          ) : (
            <div className="space-y-4">
              {filteredLessonPlans.map((plan) => (
                <div
                  key={plan.id}
                  onClick={() => handleSelectLessonPlan(plan)}
                  className={cn(
                    "border rounded-lg p-4 cursor-pointer transition-all",
                    selectedLessonPlan?.id === plan.id
                      ? "border-blue-500 bg-blue-50 shadow-sm"
                      : "border-gray-200 hover:border-blue-300 hover:bg-gray-50"
                  )}
                >
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="text-lg font-medium text-gray-900">{plan.title}</h3>
                    <div className="flex gap-2">
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                        {plan.gradeLevel}
                      </span>
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                        {plan.subject}
                      </span>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 line-clamp-3 whitespace-pre-line">
                    {plan.content.substring(0, 200)}
                    {plan.content.length > 200 ? '...' : ''}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
        
        {/* Modal Footer */}
        <div className="p-4 border-t border-gray-200 flex justify-between items-center">
          <div className="text-sm text-gray-500">
            {filteredLessonPlans.length} lesson plan{filteredLessonPlans.length !== 1 ? 's' : ''} found
          </div>
          <div className="flex gap-3">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button
              disabled={!selectedLessonPlan}
              onClick={handleConfirmSelection}
            >
              Use Selected Plan
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};