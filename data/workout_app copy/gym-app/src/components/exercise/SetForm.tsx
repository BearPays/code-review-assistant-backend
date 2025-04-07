import React from "react";
import Button from "@/components/ui/Button";

interface SetFormProps {
  setNumber: number;
  reps: number;
  weight: number;
  onRepsChange: (value: number) => void;
  onWeightChange: (value: number) => void;
  onRemove: () => void;
}

const SetForm: React.FC<SetFormProps> = ({
  setNumber,
  reps,
  weight,
  onRepsChange,
  onWeightChange,
  onRemove,
}) => {
  return (
    <div className="flex gap-3 items-center">
      <span className="w-10">#{setNumber}</span>
      <div>
        <label className="text-xs block">Reps</label>
        <input
          type="number"
          min="1"
          value={reps}
          onChange={(e) => onRepsChange(e.target.value ? parseInt(e.target.value) : 0)}
          className="w-20 p-1 border border-gray-300 rounded"
        />
      </div>
      <div>
        <label className="text-xs block">Weight (kg)</label>
        <input
          type="number"
          min="0"
          step="0.5"
          value={weight}
          onChange={(e) => onWeightChange(e.target.value ? parseFloat(e.target.value) : 0)}
          className="w-20 p-1 border border-gray-300 rounded"
        />
      </div>
      <Button
        variant="danger"
        onClick={onRemove}
        className="text-sm ml-auto"
      >
        Remove
      </Button>
    </div>
  );
};

export default SetForm;
