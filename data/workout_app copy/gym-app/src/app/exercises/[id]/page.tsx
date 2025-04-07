"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import Button from "@/components/ui/Button";
import Loading from "@/components/ui/Loading";
import ExerciseImage from "@/components/exercise/ExerciseImage";

type Exercise = {
  id: string;
  name: string;
  category: string;
  force?: string;
  level?: string;
  mechanic?: string;
  equipment?: string;
  primaryMuscles: string[];
  secondaryMuscles: string[];
  instructions: string[];
  image?: string;
};

export default function ExerciseDetailPage() {
  const { id } = useParams();
  const [exercise, setExercise] = useState<Exercise | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    if (!id) return;

    const fetchExercise = async () => {
      try {
        const res = await fetch(`/api/exercises/${id}`);
        if (res.ok) {
          const data = await res.json();
          setExercise(data);
        } else {
          throw new Error("Failed to fetch exercise details");
        }
      } catch (error) {
        console.error("Failed to fetch exercise:", error);
        // Use placeholder data if API fails
        setExercise({
          id: Array.isArray(id) ? id[0] : id,
          name: "Bench Press",
          category: "Strength",
          force: "Push",
          level: "Intermediate",
          mechanic: "Compound",
          equipment: "Barbell",
          primaryMuscles: ["Chest", "Triceps"],
          secondaryMuscles: ["Shoulders", "Core"],
          instructions: [
            "Lie on a flat bench with your feet on the floor.",
            "Grip the barbell slightly wider than shoulder width.",
            "Lower the barbell to your chest, keeping your elbows at a 45-degree angle.",
            "Push the barbell up until your arms are fully extended.",
            "Repeat for the desired number of repetitions."
          ],
          image: "/exercises/bench-press.jpg"
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchExercise();
  }, [id]);

  const goBack = () => {
    router.back();
  };

  if (isLoading) {
    return <Loading text="Loading exercise details..." />;
  }

  if (!exercise) {
    return (
      <div className="min-h-screen p-8">
        <p>Exercise not found</p>
        <Button onClick={goBack} className="mt-4">Go Back</Button>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-8 pb-24">
      <Button variant="secondary" onClick={goBack} className="mb-6">
        ← Back
      </Button>

      <h1 className="text-3xl font-bold mb-4">{exercise.name}</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div>
          <ExerciseImage 
            imageUrl={exercise.image} 
            exerciseName={exercise.name} 
            className="h-80 w-full mb-6" 
          />

          <div className="grid grid-cols-2 gap-4 mb-6">
            <div className="p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
              <h3 className="font-medium text-gray-700 dark:text-gray-300">Category</h3>
              <p className="font-semibold">{exercise.category}</p>
            </div>
            {exercise.equipment && (
              <div className="p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
                <h3 className="font-medium text-gray-700 dark:text-gray-300">Equipment</h3>
                <p className="font-semibold">{exercise.equipment}</p>
              </div>
            )}
            {exercise.level && (
              <div className="p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
                <h3 className="font-medium text-gray-700 dark:text-gray-300">Level</h3>
                <p className="font-semibold">{exercise.level}</p>
              </div>
            )}
            {exercise.mechanic && (
              <div className="p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
                <h3 className="font-medium text-gray-700 dark:text-gray-300">Mechanic</h3>
                <p className="font-semibold">{exercise.mechanic}</p>
              </div>
            )}
          </div>

          <div className="mb-6">
            <h2 className="text-xl font-bold mb-2">Target Muscles</h2>
            <div className="mb-3">
              <h3 className="font-medium text-gray-700 dark:text-gray-300">Primary</h3>
              <div className="flex flex-wrap gap-2 mt-1">
                {exercise.primaryMuscles.map(muscle => (
                  <span key={muscle} className="px-3 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-100 rounded-full text-sm">
                    {muscle}
                  </span>
                ))}
              </div>
            </div>
            {exercise.secondaryMuscles.length > 0 && (
              <div>
                <h3 className="font-medium text-gray-700 dark:text-gray-300">Secondary</h3>
                <div className="flex flex-wrap gap-2 mt-1">
                  {exercise.secondaryMuscles.map(muscle => (
                    <span key={muscle} className="px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-100 rounded-full text-sm">
                      {muscle}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        <div>
          <h2 className="text-xl font-bold mb-4">Instructions</h2>
          <ol className="list-decimal list-inside space-y-3 pl-4">
            {exercise.instructions.map((step, index) => (
              <li key={index} className="text-gray-800 dark:text-gray-200">
                {step}
              </li>
            ))}
          </ol>
        </div>
      </div>
    </div>
  );
}