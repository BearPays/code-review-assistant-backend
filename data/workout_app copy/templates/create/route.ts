import { PrismaClient } from "@prisma/client";
import { NextResponse } from "next/server";
import type { NextApiRequest, NextApiResponse } from "next";

const prisma = new PrismaClient();

interface Set {
  order: number;
  reps: number;
  weight: number;
}

interface Exercise {
  exerciseId: string;
  sets: Set[];
}

interface RequestBody {
  userId: string;
  name: string;
  exercises: Exercise[];
}

export async function POST(req: Request) {
  const { userId, name, exercises }: RequestBody = await req.json();

  if (!userId || !name || !exercises) return NextResponse.json({ error: "Missing fields" }, { status: 400 });

  try {
    const workoutTemplate = await prisma.workoutTemplate.create({
      data: {
        name,
        userId,
        exercises: {
          create: exercises.map((exercise) => ({
            exerciseId: exercise.exerciseId,
            sets: {
              create: exercise.sets.map((set) => ({
                order: set.order,
                reps: set.reps,
                weight: set.weight,
              })),
            },
          })),
        },
      },
      include: { exercises: { include: { sets: true } } },
    });

    return NextResponse.json(workoutTemplate, { status: 201 });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: "Error creating workout template" }, { status: 500 });
  }
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  const response = await POST(req);
  res.status(response.status).json(await response.json());
}
