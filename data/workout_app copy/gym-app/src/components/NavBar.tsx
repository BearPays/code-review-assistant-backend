"use client";

import Link from "next/link";
import { useAuth } from "@/contexts/AuthContext";

export default function NavBar() {
  const { isAuthenticated } = useAuth();

  return (
    <nav className="fixed bottom-0 left-0 right-0 bg-blue-800 flex justify-around items-center h-16 w-full">
      <Link href="/templates" className="nav-link">
        <div className="w-full h-full flex items-center justify-center">
          Templates
        </div>
      </Link>
      <Link href="/workouts" className="nav-link">
        <div className="w-full h-full flex items-center justify-center">
          Workouts
        </div>
      </Link>
      <Link href={isAuthenticated ? "/user" : "/login"} className="nav-link">
        <div className="w-full h-full flex items-center justify-center">
          {isAuthenticated ? "User" : "Log In"}
        </div>
      </Link>
    </nav>
  );
}
