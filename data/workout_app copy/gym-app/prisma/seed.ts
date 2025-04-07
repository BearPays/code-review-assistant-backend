import { PrismaClient } from "@prisma/client";
import fs from "fs";

const prisma = new PrismaClient();
const data = JSON.parse(fs.readFileSync("prisma/exercise_db.json", "utf-8"));

async function seed() {
  try {
    for (const exercise of data) {
      await prisma.exercise.create({
        data: {
          name: exercise.name,
          category: exercise.category,
          force: exercise.force,
          level: exercise.level,
          mechanic: exercise.mechanic,
          equipment: exercise.equipment,
          primaryMuscles: exercise.primaryMuscles || [],
          secondaryMuscles: exercise.secondaryMuscles || [],
          instructions: exercise.instructions || [],
          image: exercise.images?.length > 0
            ? `https://raw.githubusercontent.com/yuhonas/free-exercise-db/main/exercises/${exercise.images[0]}`
            : null,
        },
      });
    }

    console.log("✅ Database seeded successfully with images!");
  } catch (error) {
    console.error("❌ Error seeding database:", error);
  } finally {
    await prisma.$disconnect();
  }
}

seed();