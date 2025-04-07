import React from "react";
import Button from "@/components/ui/Button";
import { WorkoutSet } from "@/contexts/WorkoutContext";

interface WorkoutSetItemProps {
  set: WorkoutSet;
  setNumber: number;
  onUpdateSet: (field: keyof WorkoutSet, value: number | boolean) => void;
  onRemoveSet: () => void;
}

const WorkoutSetItem: React.FC<WorkoutSetItemProps> = ({
  set,
  setNumber,
  onUpdateSet,
  onRemoveSet,
}) => {
  return (
    <div className="grid grid-cols-12 items-center py-2">
      <div className="col-span-1 font-medium">{setNumber}</div>
      
      <div className="col-span-3">
        <input
          type="number"
          min="0"
          step="1.25"
          value={set.weight}
          onChange={(e) => onUpdateSet('weight', parseFloat(e.target.value) || 0)}
          className="w-full p-1 border border-gray-300 rounded text-center"
        />
      </div>
      
      <div className="col-span-3">
        <input
          type="number"
          min="0"
          value={set.reps}
          onChange={(e) => onUpdateSet('reps', parseInt(e.target.value) || 0)}
          className="w-full p-1 border border-gray-300 rounded text-center"
        />
      </div>
      
      <div className="col-span-3 text-center">
        <input
          type="checkbox"
          checked={set.completed}
          onChange={(e) => onUpdateSet('completed', e.target.checked)}
          className="h-5 w-5 accent-blue-600"
        />
      </div>
      
      <div className="col-span-2 text-right">
        <Button 
          variant="danger" 
          onClick={onRemoveSet}
          className="text-sm py-1"
        >
          Remove
        </Button>
      </div>
    </div>
  );
};

export default WorkoutSetItem;
