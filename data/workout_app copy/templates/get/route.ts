import { PrismaClient } from "@prisma/client";
import { NextResponse } from "next/server";

const prisma = new PrismaClient();

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const userId = searchParams.get("userId");

  if (!userId) return NextResponse.json({ error: "Missing userId" }, { status: 400 });

  try {
    const workouts = await prisma.workoutTemplate.findMany({
      where: { userId: userId.toString() },
      include: { exercises: { include: { sets: true } } },
    });

    return NextResponse.json(workouts, { status: 200 });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: "Error fetching workout templates" }, { status: 500 });
  }
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "GET") return res.status(405).json({ error: "Method not allowed" });

  return GET(req, res);
}
