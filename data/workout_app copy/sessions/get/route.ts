import { PrismaClient } from "@prisma/client";
import { NextResponse } from "next/server";

const prisma = new PrismaClient();

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const userId = searchParams.get("userId");

  if (!userId) return NextResponse.json({ error: "Missing userId" }, { status: 400 });

  try {
    const sessions = await prisma.workoutSession.findMany({
      where: { userId: userId.toString() },
      include: { exercises: { include: { sets: true } } },
    });

    return NextResponse.json(sessions, { status: 200 });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: "Error fetching workout sessions" }, { status: 500 });
  }
}
