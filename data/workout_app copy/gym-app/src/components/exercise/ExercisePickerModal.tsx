"use client";

import { useState, useEffect } from "react";
import Button from "@/components/ui/Button";
import Modal from "@/components/ui/Modal";
import ExercisePicker, { ExercisePickerOption } from "./ExercisePicker";

interface ExercisePickerModalProps {
  onSelectExercise: (exercise: ExercisePickerOption) => void;
  buttonText?: string;
  buttonClassName?: string;
  buttonVariant?: "primary" | "secondary" | "success" | "danger" | "outline";
  modalTitle?: string;
  providedExercises?: ExercisePickerOption[];
  overlayOpacity?: number; // Control the opacity of the modal overlay
}

const ExercisePickerModal: React.FC<ExercisePickerModalProps> = ({
  onSelectExercise,
  buttonText = "Add Exercise",
  buttonClassName = "",
  buttonVariant = "primary",
  modalTitle = "Select Exercise",
  providedExercises,
  overlayOpacity = 25, // Lighter default opacity (25%)
}) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  // Store exercises locally to avoid dependency issues
  const [localExercises, setLocalExercises] = useState<ExercisePickerOption[]>([]);
  
  // Update local exercises when providedExercises changes
  useEffect(() => {
    if (providedExercises) {
      setLocalExercises(providedExercises);
    }
  }, [providedExercises]);

  const handleOpenModal = (e: React.MouseEvent) => {
    e.preventDefault(); // Prevent any form submission
    e.stopPropagation(); // Stop event bubbling
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
  };

  const handleSelectExercise = (exercise: ExercisePickerOption) => {
    // Make a copy of the exercise to avoid reference issues
    const exerciseCopy = { ...exercise };
    onSelectExercise(exerciseCopy);
    handleCloseModal();
  };

  // Prevent click inside modal from bubbling up
  const handleModalClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  return (
    <>
      <Button
        onClick={handleOpenModal}
        variant={buttonVariant}
        className={buttonClassName}
        type="button" // Explicitly set type to prevent form submission
      >
        {buttonText}
      </Button>

      <Modal 
        isOpen={isModalOpen} 
        onClose={handleCloseModal}
        title={modalTitle}
        maxWidth="lg"
        overlayOpacity={overlayOpacity}
      >
        <div onClick={handleModalClick}>
          <ExercisePicker
            providedExercises={localExercises.length > 0 ? localExercises : providedExercises}
            onSelectExercise={handleSelectExercise}
            buttonText="Select"
          />
        </div>
      </Modal>
    </>
  );
};

export default ExercisePickerModal;
