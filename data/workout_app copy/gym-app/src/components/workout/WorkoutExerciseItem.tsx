import React from "react";
import Button from "@/components/ui/Button";
import WorkoutSetItem from "./WorkoutSetItem";
import ExerciseInfoButton from "@/components/exercise/ExerciseInfoButton";
import { WorkoutExercise, WorkoutSet } from "@/contexts/WorkoutContext";

interface WorkoutExerciseItemProps {
  exercise: WorkoutExercise;
  exerciseIndex: number; // Added this prop
  onAddSet: () => void;
  onRemoveExercise: () => void;
  onRemoveSet: (setIndex: number) => void;
  onUpdateSet: (setIndex: number, field: keyof WorkoutSet, value: number | boolean) => void;
}

const WorkoutExerciseItem: React.FC<WorkoutExerciseItemProps> = ({
  exercise,
  // exerciseIndex not used in this component
  onAddSet,
  onRemoveExercise,
  onRemoveSet,
  onUpdateSet,
}) => {
  return (
    <div className="border border-gray-300 dark:border-gray-700 rounded-lg p-4 mb-4">
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center">
          <h3 className="text-lg font-semibold">{exercise.name}</h3>
          <ExerciseInfoButton exerciseId={exercise.exerciseId} className="ml-2" />
        </div>
        <Button variant="danger" onClick={onRemoveExercise} className="text-sm py-1">
          Remove Exercise
        </Button>
      </div>
      
      <div className="mb-4">
        <div className="grid grid-cols-12 font-medium text-sm border-b pb-2 mb-2">
          <div className="col-span-1">Set</div>
          <div className="col-span-3">Weight (kg)</div>
          <div className="col-span-3">Reps</div>
          <div className="col-span-3">Completed</div>
          <div className="col-span-2"></div>
        </div>
        
        {exercise.sets.map((set, setIndex) => (
          <WorkoutSetItem
            key={set.id}
            set={set}
            setNumber={setIndex + 1}
            onUpdateSet={(field, value) => onUpdateSet(setIndex, field, value)}
            onRemoveSet={() => onRemoveSet(setIndex)}
          />
        ))}
      </div>
      
      <Button 
        variant="secondary" 
        onClick={onAddSet} 
        className="w-full"
      >
        + Add Set
      </Button>
    </div>
  );
};

export default WorkoutExerciseItem;
