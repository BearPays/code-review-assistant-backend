import Image from "next/image";

export default function Home() {
  return (
    <div className="min-h-screen p-8 pb-20 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <div className="flex items-center gap-4">
        <Image
          className="dark:invert"
          src="/next.svg"
          alt="Next.js logo"
          width={180}
          height={38}
          priority
        />
      </div>
      <main className="flex flex-col gap-8 items-center sm:items-start">
        {/* Add content here */}
      </main>
    </div>
  );
}
