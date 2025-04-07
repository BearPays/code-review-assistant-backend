"use client";
import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { useAuth } from "@/contexts/AuthContext";
import Button from "@/components/ui/Button";
import Loading from "@/components/ui/Loading";
import Link from "next/link";

type WorkoutSet = {
  reps: number;
  weight: number;
};

type WorkoutExercise = {
  id: string;
  name: string;
  sets: WorkoutSet[];
};

type WorkoutSession = {
  id: string;
  templateName: string;
  startTime: string;
  endTime: string | null;
  exercises: WorkoutExercise[];
};

export default function WorkoutDetail() {
  const params = useParams();
  const id = params.id as string; // cast id to string since we know it exists in this route
  const [workout, setWorkout] = useState<WorkoutSession | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const { isAuthenticated } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isAuthenticated) {
      router.push("/login");
      return;
    }

    const fetchWorkout = async () => {
      try {
        // In a real app, you'd fetch this data from the API
        // For this demo, we'll simulate the API response
        const res = await fetch(`/api/workouts/${id}`);
        if (res.ok) {
          const data = await res.json();
          setWorkout(data);
        } else {
          // For demo purposes, create mock data
          setWorkout({
            id: id,
            templateName: "Sample Workout",
            startTime: new Date(Date.now() - 60 * 60 * 1000).toISOString(),
            endTime: new Date().toISOString(),
            exercises: [
              {
                id: "1",
                name: "Bench Press",
                sets: [
                  { reps: 10, weight: 60 },
                  { reps: 8, weight: 65 },
                  { reps: 8, weight: 65 },
                ],
              },
              {
                id: "2",
                name: "Pull Ups",
                sets: [
                  { reps: 8, weight: 0 },
                  { reps: 6, weight: 0 },
                ],
              },
            ]
          });
        }
      } catch (error) {
        console.error("Failed to fetch workout:", error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchWorkout();
  }, [id, isAuthenticated, router]);

  if (isLoading) {
    return <Loading text="Loading workout details..." />;
  }

  if (!workout) {
    return (
      <div className="min-h-screen p-8">
        <p>Workout not found</p>
        <Link href="/workouts">
          <Button className="mt-4">Back to Workouts</Button>
        </Link>
      </div>
    );
  }

  // Calculate workout duration
  let duration = "N/A";
  if (workout.startTime && workout.endTime) {
    const start = new Date(workout.startTime).getTime();
    const end = new Date(workout.endTime).getTime();
    const diffMinutes = Math.round((end - start) / 60000);
    
    const hours = Math.floor(diffMinutes / 60);
    const minutes = diffMinutes % 60;
    
    duration = hours > 0 
      ? `${hours} hr ${minutes} min` 
      : `${minutes} min`;
  }

  // Calculate total volume (weight × reps)
  const totalVolume = workout.exercises.reduce((total, exercise) => {
    return total + exercise.sets.reduce((exerciseTotal, set) => {
      return exerciseTotal + (set.weight * set.reps);
    }, 0);
  }, 0);

  return (
    <div className="min-h-screen p-8">
      <div className="flex items-center mb-6">
        <Link href="/workouts" className="mr-4">
          <Button variant="secondary">← Back</Button>
        </Link>
        <h1 className="text-2xl font-bold">{workout.templateName}</h1>
      </div>

      <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg mb-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Date</p>
            <p>{new Date(workout.startTime).toLocaleDateString()}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Duration</p>
            <p>{duration}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Volume</p>
            <p>{totalVolume} kg</p>
          </div>
        </div>
      </div>
      
      <div>
        <h2 className="text-xl font-semibold mb-4">Exercises</h2>
        
        {workout.exercises.map((exercise) => (
          <div key={exercise.id} className="border border-gray-300 dark:border-gray-700 rounded-lg p-4 mb-4">
            <h3 className="font-semibold text-lg mb-3">{exercise.name}</h3>
            
            <table className="w-full">
              <thead>
                <tr className="text-left text-sm">
                  <th className="pb-2 w-16">Set</th>
                  <th className="pb-2">Weight</th>
                  <th className="pb-2">Reps</th>
                </tr>
              </thead>
              <tbody>
                {exercise.sets.map((set, index) => (
                  <tr key={index}>
                    <td className="py-1">{index + 1}</td>
                    <td className="py-1">{set.weight} kg</td>
                    <td className="py-1">{set.reps}</td>
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
