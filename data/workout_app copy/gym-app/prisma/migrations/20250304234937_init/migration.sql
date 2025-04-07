-- CreateTable
CREATE TABLE "Exercise" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "category" TEXT NOT NULL,
    "force" TEXT,
    "level" TEXT,
    "mechanic" TEXT,
    "equipment" TEXT,
    "primaryMuscles" TEXT[],
    "secondaryMuscles" TEXT[],
    "instructions" TEXT[],
    "image" TEXT,

    CONSTRAINT "Exercise_pkey" PRIMARY KEY ("id")
);
