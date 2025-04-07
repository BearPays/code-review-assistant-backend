"use client";

import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

export default function User() {
  const { user, logout, isAuthenticated } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isAuthenticated) {
      router.push("/login");
    }
  }, [isAuthenticated, router]);

  if (!user) {
    return <div>Loading...</div>;
  }

  return (
    <div className="min-h-screen p-8">
      <h1 className="text-2xl mb-4">User Profile</h1>
      <div className="mb-4">
        <h2 className="font-bold">Name:</h2>
        <p>{user.name}</p>
      </div>
      <div className="mb-4">
        <h2 className="font-bold">Email:</h2>
        <p>{user.email}</p>
      </div>
      <button 
        onClick={() => {
          logout();
          router.push("/");
        }} 
        className="p-2 bg-red-500 text-white rounded"
      >
        Log Out
      </button>
    </div>
  );
}
