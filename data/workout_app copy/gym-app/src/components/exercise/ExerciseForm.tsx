import React from "react";
import Button from "@/components/ui/Button";
import SetForm from "../exercise/SetForm";

interface Set {
  id: string;
  reps: number;
  weight: number;
}

interface Exercise {
  id: string;
  name: string;
  category?: string;
  sets: Set[];
}

interface ExerciseFormProps {
  exercise: Exercise;
  onRemove: () => void;
  onAddSet: () => void;
  onRemoveSet: (setIndex: number) => void;
  onSetChange: (setIndex: number, field: "reps" | "weight", value: number) => void;
}

const ExerciseForm: React.FC<ExerciseFormProps> = ({
  exercise,
  onRemove,
  onAddSet,
  onRemoveSet,
  onSetChange,
}) => {
  return (
    <div className="border border-gray-300 rounded p-4">
      <div className="flex justify-between mb-2">
        <h3 className="font-medium">{exercise.name}</h3>
        <Button 
          variant="danger" 
          onClick={onRemove}
          className="text-sm py-1 px-2"
        >
          Remove
        </Button>
      </div>

      {/* Sets */}
      <div className="space-y-3">
        {exercise.sets.map((set, setIndex) => (
          <SetForm
            key={set.id}
            setNumber={setIndex + 1}
            reps={set.reps}
            weight={set.weight}
            onRepsChange={(value) => onSetChange(setIndex, "reps", value)}
            onWeightChange={(value) => onSetChange(setIndex, "weight", value)}
            onRemove={() => onRemoveSet(setIndex)}
          />
        ))}
      </div>

      <Button
        variant="primary"
        onClick={onAddSet}
        className="mt-3 text-sm py-1 px-2"
      >
        + Add Set
      </Button>
    </div>
  );
};

export default ExerciseForm;
