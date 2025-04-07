"use client";

import { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useWorkout } from "@/contexts/WorkoutContext";
import { useRouter } from "next/navigation";
import Button from "@/components/ui/Button";
import Loading from "@/components/ui/Loading";
import Link from "next/link";

type WorkoutSession = {
  id: string;
  templateName: string;
  startTime: string;
  endTime: string | null;
  isActive: boolean;
  exercises: {
    id: string;
    name: string;
    sets: {
      reps: number;
      weight: number;
    }[];
  }[];
};

export default function Workouts() {
  const [workouts, setWorkouts] = useState<WorkoutSession[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const { isAuthenticated } = useAuth();
  const { activeWorkout } = useWorkout();
  const router = useRouter();

  useEffect(() => {
    if (!isAuthenticated) {
      router.push("/login");
      return;
    }

    const fetchWorkouts = async () => {
      try {
        const res = await fetch("/api/workouts");
        if (res.ok) {
          const data = await res.json();
          setWorkouts(data);
        } else {
          throw new Error("Failed to fetch workouts");
        }
      } catch (error) {
        console.error("Failed to fetch workouts:", error);
        // For demo, use empty data
        setWorkouts([]);
      } finally {
        setIsLoading(false);
      }
    };

    fetchWorkouts();
  }, [isAuthenticated, router]);

  if (isLoading) {
    return <Loading text="Loading workouts..." />;
  }

  const activeWorkoutBanner = activeWorkout && (
    <div className="bg-green-100 dark:bg-green-900 p-4 rounded-lg mb-6">
      <div className="flex justify-between items-center">
        <div>
          <h3 className="font-bold">Active Workout</h3>
          <p className="text-sm">{activeWorkout.templateName || 'Custom Workout'}</p>
          <p className="text-sm">Started {new Date(activeWorkout.startTime).toLocaleTimeString()}</p>
        </div>
        <Link href="/workouts/active">
          <Button variant="success">Continue Workout</Button>
        </Link>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen p-8">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Workouts</h1>
        <Link href="/workouts/new">
          <Button>Start New Workout</Button>
        </Link>
      </div>

      {activeWorkoutBanner}

      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4">Workout History</h2>
        
        {workouts.length === 0 ? (
          <div className="text-center py-10 border border-dashed border-gray-300 rounded-lg">
            <p className="mb-4">No workout history yet.</p>
            <Link href="/workouts/new">
              <Button>Start Your First Workout</Button>
            </Link>
          </div>
        ) : (
          <div className="space-y-4">
            {workouts.map((workout) => (
              <Link key={workout.id} href={`/workouts/${workout.id}`}>
                <div className="border border-gray-300 dark:border-gray-700 p-4 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800">
                  <div className="flex justify-between items-center">
                    <div>
                      <h3 className="font-semibold">{workout.templateName}</h3>
                      <p className="text-sm text-gray-600 dark:text-gray-300">
                        {new Date(workout.startTime).toLocaleDateString()}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm">
                        {workout.exercises.length} exercises
                      </p>
                      <p className="text-sm">
                        {workout.endTime 
                          ? `Duration: ${Math.round((new Date(workout.endTime).getTime() - new Date(workout.startTime).getTime()) / 60000)} min` 
                          : 'In Progress'}
                      </p>
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
