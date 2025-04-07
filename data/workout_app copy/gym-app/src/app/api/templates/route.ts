import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { cookies } from "next/headers";

// Helper function to get user ID from cookie/session
async function getUserId() {
  const userCookie = (await cookies()).get('user')?.value;
  if (!userCookie) return null;
  
  try {
    const userData = JSON.parse(userCookie);
    // In a real app, you'd verify this with a proper session check
    // For this demo, we'll find the user by email
    const user = await prisma.user.findUnique({
      where: { email: userData.email }
    });
    return user?.id;
  } catch {
    return null;
  }
}

// GET /api/templates - Get all templates for the logged-in user
export async function GET() {
  try {
    const userId = await getUserId();
    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    
    // Fetch templates for this specific user
    const templates = await prisma.workoutTemplate.findMany({
      where: { userId },
      select: {
        id: true,
        name: true,
        createdAt: true,
      },
      orderBy: { createdAt: 'desc' }
    });
    
    return NextResponse.json(templates);
  } catch (error) {
    console.error("Error fetching templates:", error);
    return NextResponse.json({ error: "Failed to fetch templates" }, { status: 500 });
  }
}

// POST /api/templates - Create a new workout template
export async function POST(request: Request) {
  try {
    const userId = await getUserId();
    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    
    const body = await request.json();
    
    // Validate request body
    if (!body.name || !body.exercises || !Array.isArray(body.exercises)) {
      return NextResponse.json(
        { error: "Invalid template data" },
        { status: 400 }
      );
    }
    
    // Create the template and related records in a transaction
    const result = await prisma.$transaction(async (tx) => {
      // 1. Create the workout template
      const workoutTemplate = await tx.workoutTemplate.create({
        data: {
          name: body.name,
          userId: userId,
        }
      });
      
      // 2. Create exercises and sets
      for (const exercise of body.exercises) {
        if (!exercise.id) continue;
        
        const workoutTemplateExercise = await tx.workoutTemplateExercise.create({
          data: {
            workoutTemplateId: workoutTemplate.id,
            exerciseId: exercise.id,
          }
        });
        
        // 3. Create sets for each exercise
        if (Array.isArray(exercise.sets)) {
          for (let i = 0; i < exercise.sets.length; i++) {
            const set = exercise.sets[i];
            await tx.workoutTemplateSet.create({
              data: {
                workoutTemplateExerciseId: workoutTemplateExercise.id,
                order: i + 1,
                reps: set.reps,
                weight: set.weight,
              }
            });
          }
        }
      }
      
      return workoutTemplate;
    });
    
    return NextResponse.json(result, { status: 201 });
  } catch (error) {
    console.error("Error creating template:", error);
    return NextResponse.json(
      { error: "Failed to create template" },
      { status: 500 }
    );
  }
}
