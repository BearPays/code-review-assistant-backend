import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function GET(request: NextRequest) {
  try {
    // Parse the ID from the URL path
    const url = new URL(request.url);
    const segments = url.pathname.split("/");
    const id = segments[segments.length - 1]; // last segment
    
    if (!id) {
      return NextResponse.json({ error: "Missing exercise ID" }, { status: 400 });
    }
    
    const exercise = await prisma.exercise.findUnique({
      where: { id },
    });

    if (!exercise) {
      return NextResponse.json({ error: "Exercise not found" }, { status: 404 });
    }

    return NextResponse.json(exercise);
  } catch (error) {
    console.error("Error fetching exercise details:", error);
    return NextResponse.json(
      { error: "Failed to fetch exercise details" },
      { status: 500 }
    );
  }
}