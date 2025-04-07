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

// GET /api/templates/[id] - Get a specific template
export async function GET(request: NextRequest): Promise<NextResponse> {
  try {
    const userId = await getUserId();
    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    
    const url = new URL(request.url);
    const segments = url.pathname.split('/');
    const id = segments[segments.length - 1];
    
    // Get the template with exercises and sets
    const template = await prisma.workoutTemplate.findUnique({
      where: { id },
      include: {
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
    
    if (!template) {
      return NextResponse.json({ error: "Template not found" }, { status: 404 });
    }
    
    if (template.userId !== userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 403 });
    }
    
    // Transform the data to match the expected format
    const formattedTemplate = {
      id: template.id,
      name: template.name,
      createdAt: template.createdAt.toISOString(),
      exercises: template.exercises.map(ex => ({
        id: ex.exerciseId,
        name: ex.exercise.name,
        sets: ex.sets.map(set => ({
          id: set.id,
          reps: set.reps,
          weight: set.weight
        }))
      }))
    };
    
    return NextResponse.json(formattedTemplate);
  } catch (error) {
    console.error(`Error fetching template:`, error);
    return NextResponse.json(
      { error: "Failed to fetch template" },
      { status: 500 }
    );
  }
}

// PUT /api/templates/[id] - Update a specific template
export async function PUT(request: NextRequest): Promise<NextResponse> {
  try {
    const userId = await getUserId();
    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    
    const url = new URL(request.url);
    const segments = url.pathname.split('/');
    const templateId = segments[segments.length - 1];
    const body = await request.json();
    
    // Validate request body
    if (!body.name || !body.exercises || !Array.isArray(body.exercises)) {
      return NextResponse.json(
        { error: "Invalid template data" },
        { status: 400 }
      );
    }
    
    // Verify the template belongs to this user
    const existingTemplate = await prisma.workoutTemplate.findUnique({
      where: { id: templateId },
      select: { userId: true }
    });

    if (!existingTemplate) {
      return NextResponse.json({ error: "Template not found" }, { status: 404 });
    }
    
    if (existingTemplate.userId !== userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 403 });
    }
    
    // Fetch exercises separately
    const existingExercises = await prisma.workoutTemplateExercise.findMany({
      where: { workoutTemplateId: templateId },
      include: {
        sets: true
      }
    });
    
    // Update in a transaction to ensure consistency
    await prisma.$transaction(async (tx) => {
      // Update template name
      await tx.workoutTemplate.update({
        where: { id: templateId },
        data: { name: body.name }
      });
      
      // First, delete all sets for each exercise
      for (const exercise of existingExercises) {
        await tx.workoutTemplateSet.deleteMany({
          where: { workoutTemplateExerciseId: exercise.id }
        });
      }
      
      // Then delete all exercises
      await tx.workoutTemplateExercise.deleteMany({
        where: { workoutTemplateId: templateId }
      });
      
      // Create new exercises and sets
      for (const exercise of body.exercises) {
        if (!exercise.id) continue;
        
        const workoutTemplateExercise = await tx.workoutTemplateExercise.create({
          data: {
            workoutTemplateId: templateId,
            exerciseId: exercise.id,
          }
        });
        
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
    });
    
    // Get the updated template with all its relations
    const updatedTemplate = await prisma.workoutTemplate.findUnique({
      where: { id: templateId },
      include: {
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
    
    if (!updatedTemplate) {
      return NextResponse.json({ error: "Failed to retrieve updated template" }, { status: 500 });
    }
    
    // Format the response
    const formattedTemplate = {
      id: updatedTemplate.id,
      name: updatedTemplate.name,
      createdAt: updatedTemplate.createdAt.toISOString(),
      exercises: updatedTemplate.exercises.map(ex => ({
        id: ex.exerciseId,
        name: ex.exercise.name,
        sets: ex.sets.map(set => ({
          id: set.id,
          reps: set.reps,
          weight: set.weight
        }))
      }))
    };
    
    return NextResponse.json(formattedTemplate);
  } catch (error) {
    console.error(`Error updating template:`, error);
    return NextResponse.json(
      { error: "Failed to update template" },
      { status: 500 }
    );
  }
}

// DELETE /api/templates/[id] - Delete a specific template
export async function DELETE(request: NextRequest): Promise<NextResponse> {
  try {
    const userId = await getUserId();
    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    
    const url = new URL(request.url);
    const segments = url.pathname.split('/');
    const templateId = segments[segments.length - 1];
    
    // Verify the template belongs to this user
    const existingTemplate = await prisma.workoutTemplate.findUnique({
      where: { id: templateId },
      select: { userId: true }
    });
    
    if (!existingTemplate) {
      return NextResponse.json({ error: "Template not found" }, { status: 404 });
    }
    
    if (existingTemplate.userId !== userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 403 });
    }
    
    // Delete template and its related records
    await prisma.$transaction(async (tx) => {
      // First find all exercises for this template
      const exercises = await tx.workoutTemplateExercise.findMany({
        where: { workoutTemplateId: templateId }
      });
      
      // Delete sets for each exercise
      for (const exercise of exercises) {
        await tx.workoutTemplateSet.deleteMany({
          where: { workoutTemplateExerciseId: exercise.id }
        });
      }
      
      // Delete exercises
      await tx.workoutTemplateExercise.deleteMany({
        where: { workoutTemplateId: templateId }
      });
      
      // Finally delete the template
      await tx.workoutTemplate.delete({
        where: { id: templateId }
      });
    });
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error(`Error deleting template:`, error);
    return NextResponse.json(
      { error: "Failed to delete template" },
      { status: 500 }
    );
  }
}
