"use client";

import { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useWorkout, WorkoutExercise } from "@/contexts/WorkoutContext";
import { useRouter } from "next/navigation";
import Button from "@/components/ui/Button";
import Loading from "@/components/ui/Loading";
import Link from "next/link";

type WorkoutTemplate = {
  id: string;
  name: string;
  createdAt: string;
};

export default function NewWorkout() {
  const [templates, setTemplates] = useState<WorkoutTemplate[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const { isAuthenticated } = useAuth();
  const { startWorkout, activeWorkout } = useWorkout();
  const router = useRouter();

  useEffect(() => {
    if (!isAuthenticated) {
      router.push("/login");
      return;
    }

    // If already have an active workout, redirect to it
    if (activeWorkout) {
      router.push("/workouts/active");
      return;
    }

    const fetchTemplates = async () => {
      try {
        const res = await fetch("/api/templates");
        if (res.ok) {
          const data = await res.json();
          setTemplates(data);
        } else {
          throw new Error("Failed to fetch templates");
        }
      } catch (error) {
        console.error("Failed to fetch templates:", error);
        setTemplates([]);
      } finally {
        setIsLoading(false);
      }
    };

    fetchTemplates();
  }, [isAuthenticated, router, activeWorkout]);

  const handleStartEmptyWorkout = () => {
    startWorkout();
    router.push("/workouts/active");
  };

  const handleStartTemplateWorkout = async (templateId: string, templateName: string) => {
    // Fetch template details to get exercises
    try {
      const res = await fetch(`/api/templates/${templateId}`);
      if (res.ok) {
        const template = await res.json();
        
        // Convert template exercises to workout exercises
        type TemplateExercise = {
          id: string;
          name: string;
          sets: { reps: number; weight: number }[];
        };

        const workoutExercises: WorkoutExercise[] = template.exercises.map((ex: TemplateExercise) => ({
          id: Date.now().toString() + Math.random().toString(),
          name: ex.name,
          exerciseId: ex.id,
          sets: ex.sets.map((set) => ({
            id: Date.now().toString() + Math.random().toString(),
            reps: set.reps,
            weight: set.weight,
            completed: false,
          })),
        }));
        
        startWorkout(templateId, templateName, workoutExercises);
        router.push("/workouts/active");
      } else {
        throw new Error("Failed to fetch template details");
      }
    } catch (error) {
      console.error("Failed to start template workout:", error);
      alert("Failed to start workout from template");
    }
  };

  if (isLoading) {
    return <Loading text="Loading templates..." />;
  }

  return (
    <div className="min-h-screen p-8">
      <div className="flex items-center mb-8">
        <Link href="/workouts" className="mr-4">
          <Button variant="secondary">← Back</Button>
        </Link>
        <h1 className="text-2xl font-bold">Start New Workout</h1>
      </div>

      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4">Empty Workout</h2>
        <div className="border border-gray-300 dark:border-gray-700 p-4 rounded-lg">
          <p className="mb-4">Start with a blank workout and add your own exercises</p>
          <Button onClick={handleStartEmptyWorkout}>Start Empty Workout</Button>
        </div>
      </div>

      <div>
        <h2 className="text-xl font-semibold mb-4">From Template</h2>
        
        {templates.length === 0 ? (
          <div className="text-center py-10 border border-dashed border-gray-300 rounded-lg">
            <p className="mb-4">No templates found. Create a template first.</p>
            <Link href="/templates/create">
              <Button variant="secondary">Create Template</Button>
            </Link>
          </div>
        ) : (
          <div className="space-y-4">
            {templates.map(template => (
              <div key={template.id} className="border border-gray-300 dark:border-gray-700 p-4 rounded-lg">
                <div className="flex justify-between items-center">
                  <div>
                    <h3 className="font-semibold">{template.name}</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                      Created {new Date(template.createdAt).toLocaleDateString()}
                    </p>
                  </div>
                  <Button onClick={() => handleStartTemplateWorkout(template.id, template.name)}>
                    Start Workout
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
