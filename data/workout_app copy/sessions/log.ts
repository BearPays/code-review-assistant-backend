import { PrismaClient } from "@prisma/client";
import { NextApiRequest, NextApiResponse } from "next";

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
  userId: number;
  workoutTemplateId?: number;
  exercises: Exercise[];
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  const { userId, exercises } = req.body as RequestBody;

  if (!userId || !exercises) return res.status(400).json({ error: "Missing fields" });

  try {
    const session = await prisma.workoutSession.create({
      data: {
        userId: userId.toString(),
        workoutTemplateId: req.body.workoutTemplateId,
        startTime: new Date(),
        exercises: {
          create: exercises.map((exercise) => ({
            exercise: { connect: { id: exercise.exerciseId } },
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

    return res.status(201).json(session);
  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: "Error logging workout" });
  }
}