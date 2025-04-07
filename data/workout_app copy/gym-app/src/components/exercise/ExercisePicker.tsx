"use client";

import { useState, useEffect, useMemo } from "react";
import Button from "@/components/ui/Button";
import Input from "@/components/ui/Input";
import Select from "@/components/ui/Select";
import Loading from "@/components/ui/Loading";
import { FiSearch } from "react-icons/fi";
import ExerciseInfoButton from "./ExerciseInfoButton";

export type ExercisePickerOption = {
  id: string;
  name: string;
  category: string;
  primaryMuscles: string[];
  equipment: string | null;
};

interface ExercisePickerProps {
  onSelectExercise: (exercise: ExercisePickerOption) => void;
  className?: string;
  buttonText?: string;
  providedExercises?: ExercisePickerOption[]; // Optional prop to receive exercises from parent
}

const ExercisePicker: React.FC<ExercisePickerProps> = ({
  onSelectExercise,
  className = "",
  buttonText = "Add Exercise",
  providedExercises
}) => {
  const [exercises, setExercises] = useState<ExercisePickerOption[]>([]);
  const [isLoading, setIsLoading] = useState(providedExercises ? false : true);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedBodyPart, setSelectedBodyPart] = useState("");
  const [selectedEquipment, setSelectedEquipment] = useState("");

  // Use the provided exercises if available, otherwise fetch them
  useEffect(() => {
    if (providedExercises) {
      setExercises(providedExercises);
      setIsLoading(false);
      return;
    }

    const fetchExercises = async () => {
      try {
        const res = await fetch("/api/exercises");
        if (res.ok) {
          const data = await res.json();
          setExercises(data);
        } else {
          throw new Error("Failed to fetch exercises");
        }
      } catch (error) {
        console.error("Failed to fetch exercises:", error);
        setExercises([]);
      } finally {
        setIsLoading(false);
      }
    };

    fetchExercises();
  }, [providedExercises]);

  // Extract unique body parts from primaryMuscles arrays
  const bodyParts = useMemo(() => {
    const parts = new Set<string>();
    exercises.forEach(exercise => {
      if (exercise.primaryMuscles) {
        exercise.primaryMuscles.forEach(muscle => parts.add(muscle));
      }
    });
    return Array.from(parts).sort();
  }, [exercises]);

  // Extract unique equipment types
  const equipmentTypes = useMemo(() => {
    const types = new Set<string>();
    exercises.forEach(exercise => {
      if (exercise.equipment) types.add(exercise.equipment);
    });
    return Array.from(types).sort();
  }, [exercises]);

  // Filter exercises based on search query and filters
  const filteredExercises = useMemo(() => {
    return exercises.filter(exercise => {
      // Filter by search query
      const matchesSearch = searchQuery === "" || 
        exercise.name.toLowerCase().includes(searchQuery.toLowerCase());
      
      // Filter by body part
      const matchesBodyPart = selectedBodyPart === "" || 
        (exercise.primaryMuscles && exercise.primaryMuscles.includes(selectedBodyPart));
      
      // Filter by equipment
      const matchesEquipment = selectedEquipment === "" || 
        exercise.equipment === selectedEquipment;
      
      return matchesSearch && matchesBodyPart && matchesEquipment;
    });
  }, [exercises, searchQuery, selectedBodyPart, selectedEquipment]);

  const handleBodyPartChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedBodyPart(e.target.value);
  };

  const handleEquipmentChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedEquipment(e.target.value);
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
  };

  const handleClearFilters = () => {
    setSearchQuery("");
    setSelectedBodyPart("");
    setSelectedEquipment("");
  };

  const handleSelectExercise = (exercise: ExercisePickerOption) => {
    onSelectExercise(exercise);
  };

  if (isLoading) {
    return <Loading text="Loading exercises..." />;
  }

  // Update the height for the modal context
  return (
    <div className={`${className}`}>
      <div className="mb-4">
        <div className="relative mb-2">
          <Input
            id="exerciseSearch"
            type="text"
            placeholder="Search exercises..."
            value={searchQuery}
            onChange={handleSearchChange}
            className="pl-10"
          />
          <FiSearch className="absolute left-3 top-3 text-gray-400" size={18} />
        </div>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mb-2">
          <Select
            id="bodyPartFilter"
            value={selectedBodyPart}
            onChange={handleBodyPartChange}
            options={[
              { value: "", label: "All Body Parts" },
              ...bodyParts.map(part => ({ value: part, label: part }))
            ]}
          />
          
          <Select
            id="equipmentFilter"
            value={selectedEquipment}
            onChange={handleEquipmentChange}
            options={[
              { value: "", label: "All Equipment" },
              ...equipmentTypes.map(type => ({ value: type, label: type }))
            ]}
          />
        </div>
        
        {(searchQuery || selectedBodyPart || selectedEquipment) && (
          <div className="flex justify-between items-center text-sm mb-2">
            <span className="text-gray-500">
              {filteredExercises.length} exercise{filteredExercises.length !== 1 ? 's' : ''} found
            </span>
            <button
              onClick={handleClearFilters}
              className="text-blue-600 hover:underline"
              type="button"
            >
              Clear Filters
            </button>
          </div>
        )}
      </div>

      <div className="max-h-80 overflow-y-auto border border-gray-300 dark:border-gray-700 rounded-lg">
        {filteredExercises.length === 0 ? (
          <div className="p-4 text-center text-gray-500">
            No exercises found matching the filters.
          </div>
        ) : (
          <div className="divide-y divide-gray-300 dark:divide-gray-700">
            {filteredExercises.map(exercise => (
              <div key={exercise.id} className="p-3 hover:bg-gray-100 dark:hover:bg-gray-800">
                <div className="flex justify-between items-center">
                  <div className="flex-1">
                    <div className="flex items-center">
                      <h4 className="font-medium">{exercise.name}</h4>
                      <ExerciseInfoButton exerciseId={exercise.id} className="ml-1" />
                    </div>
                    <div className="text-sm text-gray-500 flex flex-wrap gap-2 mt-1">
                      {exercise.category && (
                        <span>{exercise.category}</span>
                      )}
                      {exercise.equipment && (
                        <>
                          <span>•</span>
                          <span>{exercise.equipment}</span>
                        </>
                      )}
                      {exercise.primaryMuscles && exercise.primaryMuscles.length > 0 && (
                        <>
                          <span>•</span>
                          <span>{exercise.primaryMuscles.join(", ")}</span>
                        </>
                      )}
                    </div>
                  </div>
                  <Button
                    onClick={() => handleSelectExercise(exercise)}
                    size="sm"
                  >
                    {buttonText}
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ExercisePicker;
