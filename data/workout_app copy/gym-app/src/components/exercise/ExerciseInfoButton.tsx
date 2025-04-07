import React from "react";
import { useRouter } from "next/navigation";
import { FiInfo } from "react-icons/fi";

interface ExerciseInfoButtonProps {
  exerciseId: string;
  className?: string;
}

const ExerciseInfoButton: React.FC<ExerciseInfoButtonProps> = ({ exerciseId, className = "" }) => {
  const router = useRouter();

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent triggering parent click handlers
    router.push(`/exercises/${exerciseId}`);
  };

  return (
    <button 
      onClick={handleClick}
      className={`p-1 text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 ${className}`}
      title="View exercise details"
    >
      <FiInfo size={18} />
    </button>
  );
};

export default ExerciseInfoButton;
