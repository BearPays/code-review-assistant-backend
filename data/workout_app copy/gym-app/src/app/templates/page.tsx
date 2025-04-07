"use client";

import { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import Link from "next/link";
import { useRouter } from "next/navigation";
import Button from "@/components/ui/Button";
import Loading from "@/components/ui/Loading";
import TemplateCard from "@/components/templates/TemplateCard";

type WorkoutTemplate = {
  id: string;
  name: string;
  createdAt: string;
};

export default function Templates() {
  const [templates, setTemplates] = useState<WorkoutTemplate[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const { isAuthenticated } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isAuthenticated) {
      router.push("/login");
      return;
    }

    const fetchTemplates = async () => {
      try {
        const res = await fetch("/api/templates");
        if (res.ok) {
          const data = await res.json();
          setTemplates(data);
        } else {
          throw new Error("Failed to fetch templates");
        }
      } catch (error) {
        console.error("Failed to fetch templates:", error);
        setTemplates([]);
      } finally {
        setIsLoading(false);
      }
    };

    fetchTemplates();
  }, [isAuthenticated, router]);

  if (isLoading) {
    return <Loading />;
  }

  return (
    <div className="min-h-screen p-8">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Workout Templates</h1>
        <Link href="/templates/create">
          <Button>Create New</Button>
        </Link>
      </div>

      {templates.length === 0 ? (
        <div className="text-center py-10 border border-dashed border-gray-300 rounded-lg">
          <p className="mb-4">No templates found. Create your first workout template!</p>
          <Link href="/templates/create">
            <Button>Create Template</Button>
          </Link>
        </div>
      ) : (
        <div className="grid gap-4">
          {templates.map((template) => (
            <TemplateCard
              key={template.id}
              id={template.id}
              name={template.name}
              createdAt={template.createdAt}
            />
          ))}
        </div>
      )}
    </div>
  );
}
