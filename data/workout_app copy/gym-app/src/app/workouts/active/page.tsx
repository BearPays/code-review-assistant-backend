"use client";

import { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useWorkout } from "@/contexts/WorkoutContext";
import { useRouter } from "next/navigation";
import Button from "@/components/ui/Button";
import Loading from "@/components/ui/Loading";
import WorkoutExerciseItem from "@/components/workout/WorkoutExerciseItem";
import ExercisePickerModal from "@/components/exercise/ExercisePickerModal";
import { ExercisePickerOption } from "@/components/exercise/ExercisePicker";
import Link from "next/link";

export default function ActiveWorkout() {
  const [isLoading, setIsLoading] = useState(true);
  const [isFinishing, setIsFinishing] = useState(false);
  const [workoutDuration, setWorkoutDuration] = useState("00:00:00");
  const [availableExercises, setAvailableExercises] = useState<ExercisePickerOption[]>([]);
  
  const { activeWorkout, addExercise, removeExercise, addSet, removeSet, updateSet, endWorkout, isSaving } = useWorkout();
  const { isAuthenticated } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isAuthenticated) {
      router.push("/login");
      return;
    }

    if (!activeWorkout) {
      router.push("/workouts");
      return;
    }

    const fetchExercises = async () => {
      try {
        const res = await fetch("/api/exercises");
        if (res.ok) {
          const data = await res.json();
          setAvailableExercises(data);
        } else {
          throw new Error("Failed to fetch exercises");
        }
      } catch (error) {
        console.error("Failed to fetch exercises:", error);
        setAvailableExercises([]); // Set empty array on error
      } finally {
        setIsLoading(false);
      }
    };

    fetchExercises();
  }, [isAuthenticated, router, activeWorkout]);

  // Update the workout duration every second
  useEffect(() => {
    if (!activeWorkout) return;
    
    const updateDuration = () => {
      const startTime = new Date(activeWorkout.startTime).getTime();
      const now = Date.now();
      const diffMs = now - startTime;
      
      const hours = Math.floor(diffMs / (1000 * 60 * 60)).toString().padStart(2, '0');
      const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60)).toString().padStart(2, '0');
      const seconds = Math.floor((diffMs % (1000 * 60)) / 1000).toString().padStart(2, '0');
      
      setWorkoutDuration(`${hours}:${minutes}:${seconds}`);
    };
    
    updateDuration();
    const interval = setInterval(updateDuration, 1000);
    
    return () => clearInterval(interval);
  }, [activeWorkout]);

  const handleAddExercise = (exercise: ExercisePickerOption) => {
    addExercise(exercise.id, exercise.name);
  };

  const handleFinishWorkout = async () => {
    if (!activeWorkout) return;
    
    setIsFinishing(true);
    
    try {
      const success = await endWorkout();
      if (success) {
        router.push("/workouts");
      } else {
        alert("Failed to finish workout. Please try again.");
      }
    } catch (error) {
      console.error("Error finishing workout:", error);
      alert("An error occurred while finishing workout");
    } finally {
      setIsFinishing(false);
    }
  };

  if (isLoading) {
    return <Loading text="Loading workout..." />;
  }

  if (!activeWorkout) {
    return (
      <div className="min-h-screen p-8 text-center">
        <p>No active workout found.</p>
        <Link href="/workouts/new">
          <Button className="mt-4">Start a New Workout</Button>
        </Link>
      </div>
    );
  }

  const totalSets = activeWorkout.exercises.reduce(
    (sum, exercise) => sum + exercise.sets.length, 0
  );
  
  const completedSets = activeWorkout.exercises.reduce(
    (sum, exercise) => sum + exercise.sets.filter(set => set.completed).length, 0
  );

  const completionPercentage = totalSets > 0 
    ? Math.round((completedSets / totalSets) * 100) 
    : 0;

  return (
    <div className="min-h-screen p-8 pb-24">
      <div className="flex items-center mb-6">
        <Link href="/workouts" className="mr-4">
          <Button variant="secondary">← Back</Button>
        </Link>
        <h1 className="text-2xl font-bold">{activeWorkout.templateName || 'Custom Workout'}</h1>
      </div>

      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg mb-6">
        <div className="flex flex-wrap justify-between items-center">
          <div>
            <p className="text-gray-600 dark:text-gray-300">Started: {new Date(activeWorkout.startTime).toLocaleTimeString()}</p>
            <p className="text-gray-600 dark:text-gray-300">Duration: {workoutDuration}</p>
          </div>
          <div className="text-right">
            <p className="text-lg font-semibold">{completionPercentage}% Complete</p>
            <p className="text-sm">{completedSets} / {totalSets} sets</p>
          </div>
        </div>

        <div className="mt-3 bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
          <div 
            className="bg-blue-600 h-2.5 rounded-full" 
            style={{ width: `${completionPercentage}%` }}
          ></div>
        </div>
      </div>

      <div className="mb-8">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">Exercises</h2>
          <ExercisePickerModal
            providedExercises={availableExercises}
            onSelectExercise={handleAddExercise}
            buttonText="Add Exercise"
            modalTitle="Add Exercise to Workout"
          />
        </div>
        
        <p className="text-gray-600 dark:text-gray-300 mb-4">
          {activeWorkout.exercises.length} exercise{activeWorkout.exercises.length !== 1 ? 's' : ''} added
        </p>
        
        {activeWorkout.exercises.length === 0 ? (
          <div className="text-center py-10 border border-dashed border-gray-300 rounded-lg">
            <p className="mb-4">No exercises added yet. Add your first exercise to start.</p>
            <ExercisePickerModal
              providedExercises={availableExercises}
              onSelectExercise={handleAddExercise}
              buttonText="Add First Exercise"
              buttonVariant="primary"
              modalTitle="Select Exercise"
            />
          </div>
        ) : (
          <div className="space-y-6">
            {activeWorkout.exercises.map((exercise, exerciseIndex) => (
              <WorkoutExerciseItem
                key={exercise.id}
                exercise={exercise}
                exerciseIndex={exerciseIndex}
                onAddSet={() => addSet(exerciseIndex)}
                onRemoveExercise={() => removeExercise(exerciseIndex)}
                onRemoveSet={(setIndex) => removeSet(exerciseIndex, setIndex)}
                onUpdateSet={(setIndex, field, value) => 
                  updateSet(exerciseIndex, setIndex, field, value)
                }
              />
            ))}
          </div>
        )}
      </div>
      
      <div className="fixed bottom-20 left-0 right-0 p-4 bg-white dark:bg-black border-t border-gray-200 dark:border-gray-800">
        <Button
          variant="success"
          onClick={handleFinishWorkout}
          fullWidth
          disabled={isFinishing || isSaving || activeWorkout.exercises.length === 0}
        >
          {isFinishing ? "Finishing..." : "Finish Workout"}
        </Button>
      </div>
    </div>
  );
}
