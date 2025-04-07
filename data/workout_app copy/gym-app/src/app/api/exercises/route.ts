import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

// GET /api/exercises - Get all available exercises
export async function GET() {
  try {
    // Fetch exercises from the database
    const exercises = await prisma.exercise.findMany({
      select: {
        id: true,
        name: true,
        category: true,
        primaryMuscles: true,
        equipment: true,
      }
    });
    
    return NextResponse.json(exercises);
  } catch (error) {
    console.error("Error fetching exercises:", error);
    return NextResponse.json(
      { error: "Failed to fetch exercises" },
      { status: 500 }
    );
  }
}
