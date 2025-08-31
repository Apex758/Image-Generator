import React, { useState } from 'react';
import { BookOpen } from 'lucide-react';
import { Button } from '../ui/Button';
import { LessonPlanModal } from './LessonPlanModal';
import { LessonPlan } from './mockData';

interface LessonPlanButtonProps {
  onSelectLessonPlan: (lessonPlan: LessonPlan) => void;
  className?: string;
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
}

export const LessonPlanButton: React.FC<LessonPlanButtonProps> = ({
  onSelectLessonPlan,
  className,
  variant = 'primary',
  size = 'md',
}) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedLessonPlan, setSelectedLessonPlan] = useState<LessonPlan | null>(null);

  const openModal = () => {
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
  };

  const handleSelectLessonPlan = (lessonPlan: LessonPlan) => {
    setSelectedLessonPlan(lessonPlan);
    onSelectLessonPlan(lessonPlan);
  };

  return (
    <>
      <Button
        variant={variant}
        size={size}
        className={className}
        onClick={openModal}
      >
        <BookOpen className="mr-2 h-4 w-4" />
        Select Lesson Plan
      </Button>

      <LessonPlanModal
        isOpen={isModalOpen}
        onClose={closeModal}
        onSelectLessonPlan={handleSelectLessonPlan}
      />
    </>
  );
};