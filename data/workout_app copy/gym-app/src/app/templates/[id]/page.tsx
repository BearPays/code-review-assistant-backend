"use client";

import { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import Link from "next/link";
import Button from "@/components/ui/Button";
import Loading from "@/components/ui/Loading";
import ExerciseInfoButton from "@/components/exercise/ExerciseInfoButton";

type Set = {
  id: string;
  reps: number;
  weight: number;
};

type Exercise = {
  id: string;
  name: string;
  sets: Set[];
};

type WorkoutTemplate = {
  id: string;
  name: string;
  createdAt: string;
  exercises: Exercise[];
};

export default function TemplateDetailPage({ params: promisedParams }: { params: Promise<{ id: string }> }) {
  const [template, setTemplate] = useState<WorkoutTemplate | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isDeleting, setIsDeleting] = useState(false);
  const { isAuthenticated } = useAuth();
  const router = useRouter();
  const [params, setParams] = useState<{ id: string } | null>(null);

  useEffect(() => {
    promisedParams.then((unwrappedParams) => {
      setParams(unwrappedParams);
    });
  }, [promisedParams]);

  useEffect(() => {
    // Wait until params is set
    if (params === null) return;

    // Then check authentication
    if (!isAuthenticated) {
      router.push("/login");
      return;
    }

    const fetchTemplate = async () => {
      try {
        const res = await fetch(`/api/templates/${params.id}`);
        if (res.ok) {
          const data = await res.json();
          setTemplate(data);
        } else {
          throw new Error("Failed to fetch template details");
        }
      } catch (error) {
        console.error("Failed to fetch template details:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchTemplate();
  }, [params, isAuthenticated, router]);

  const handleStartWorkout = () => {
    router.push("/workouts");
  };

  const handleDeleteTemplate = async () => {
    if (!confirm("Are you sure you want to delete this template? This action cannot be undone.")) {
      return;
    }

    if (!params) {
      return;
    }

    setIsDeleting(true);
    
    try {
      const response = await fetch(`/api/templates/${params.id}`, {
        method: "DELETE",
      });

      if (response.ok) {
        router.push("/templates");
      } else {
        alert("Failed to delete template");
      }
    } catch (error) {
      console.error("Error deleting template:", error);
      alert("Failed to delete template due to a network error.");
    } finally {
      setIsDeleting(false);
    }
  };

  if (isLoading) {
    return <Loading />;
  }

  if (!template) {
    return <div className="min-h-screen p-8">Template not found</div>;
  }

  return (
    <div className="min-h-screen p-8">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">{template.name}</h1>
        <div className="flex gap-2">
          <Link href={`/templates/${params?.id}/edit`}>
            <Button>Edit</Button>
          </Link>
          <Button 
            variant="success" 
            onClick={handleStartWorkout}
          >
            Start Workout
          </Button>
          <Button 
            variant="danger" 
            onClick={handleDeleteTemplate} 
            disabled={isDeleting}
          >
            {isDeleting ? "Deleting..." : "Delete"}
          </Button>
        </div>
      </div>

      <div className="mb-4">
        <p className="text-sm text-gray-500">
          Created {new Date(template.createdAt).toLocaleDateString()}
        </p>
      </div>

      <div className="space-y-6 mt-8">
        {template.exercises.map((exercise, index) => (
          <div key={`${exercise.id}-${index}`} className="border border-gray-300 rounded p-4">
            <div className="flex items-center mb-3">
              <h3 className="font-medium">{exercise.name}</h3>
              <ExerciseInfoButton exerciseId={exercise.id} className="ml-2" />
            </div>

            <table className="w-full">
              <thead>
                <tr className="text-left text-sm">
                  <th className="pb-2 w-16">Set</th>
                  <th className="pb-2">Reps</th>
                  <th className="pb-2">Weight</th>
                </tr>
              </thead>
              <tbody>
                {exercise.sets.map((set, index) => (
                  <tr key={set.id}>
                    <td className="py-2">{index + 1}</td>
                    <td className="py-2">{set.reps}</td>
                    <td className="py-2">{set.weight} kg</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </div>
  );
}
