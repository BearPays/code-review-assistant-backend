import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { prisma } from '@/lib/prisma';

// Helper function to get user ID from cookie/session
async function getUserId() {
  const userCookies = await cookies();
  const userCookie = userCookies.get('user')?.value;
  if (!userCookie) return null;
  
  try {
    const userData = JSON.parse(userCookie);
    const user = await prisma.user.findUnique({
      where: { email: userData.email }
    });
    return user?.id;
  } catch {
    return null;
  }
}

export async function GET(request: NextRequest): Promise<NextResponse> {
  try {
    const userId = await getUserId();
    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    
    // Parse the ID from the URL path
    const url = new URL(request.url);
    const segments = url.pathname.split('/');
    const id = segments[segments.length - 1];
    
    // Get the workout with exercises and sets
    const workout = await prisma.workoutSession.findUnique({
      where: { id },
      include: {
        workoutTemplate: true,
        exercises: {
          include: {
            exercise: true,
            sets: {
              orderBy: { order: 'asc' }
            }
          }
        }
      }
    });
    
    if (!workout) {
      return NextResponse.json({ error: "Workout not found" }, { status: 404 });
    }
    
    if (workout.userId !== userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 403 });
    }
    
    // Format response
    const formattedWorkout = {
      id: workout.id,
      templateName: workout.workoutTemplate?.name || 'Custom Workout',
      startTime: workout.startTime.toISOString(),
      endTime: workout.endTime?.toISOString(),
      exercises: workout.exercises.map(ex => ({
        id: ex.id,
        name: ex.exercise.name,
        sets: ex.sets.map(set => ({
          reps: set.reps,
          weight: set.weight,
        }))
      }))
    };
    
    return NextResponse.json(formattedWorkout);
  } catch (error) {
    console.error(`Error fetching workout:`, error);
    return NextResponse.json(
      { error: "Failed to fetch workout" },
      { status: 500 }
    );
  }
}
