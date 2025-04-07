import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { cookies } from "next/headers";
import * as bcrypt from "bcryptjs";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    
    if (body.action === 'login') {
      // Login flow
      const { email, password } = body;
      
      const user = await prisma.user.findUnique({
        where: { email }
      });
      
      if (!user) {
        return NextResponse.json({ error: "User not found" }, { status: 401 });
      }
      
      // In a real app, verify password with bcrypt
      const passwordValid = await bcrypt.compare(password, user.password);
      
      if (!passwordValid) {
        return NextResponse.json({ error: "Invalid password" }, { status: 401 });
      }
      
      // Set a cookie with user info
      (await cookies()).set('user', JSON.stringify({ 
        email: user.email,
        name: user.name || email.split('@')[0]
      }));
      
      return NextResponse.json({ 
        success: true,
        user: { 
          email: user.email,
          name: user.name || email.split('@')[0]
        }
      });
      
    } else if (body.action === 'register') {
      // Registration flow
      const { name, email, password } = body;
      
      // Check if user exists
      const existingUser = await prisma.user.findUnique({
        where: { email }
      });
      
      if (existingUser) {
        return NextResponse.json({ error: "Email already in use" }, { status: 400 });
      }
      
      // Create new user
      // In a real app, hash the password
      // const hashedPassword = await bcrypt.hash(password, 10);
      const hashedPassword = password; // For demo
      
      const user = await prisma.user.create({
        data: {
          email,
          name,
          password: hashedPassword
        }
      });
      
      // Set a cookie with user info
      (await cookies()).set('user', JSON.stringify({ 
        email: user.email,
        name: user.name || email.split('@')[0]
      }));
      
      return NextResponse.json({ 
        success: true,
        user: { 
          email: user.email,
          name: user.name || email.split('@')[0]
        }
      }, { status: 201 });
    }
    
    return NextResponse.json({ error: "Invalid action" }, { status: 400 });
  } catch (error) {
    console.error("Auth error:", error);
    return NextResponse.json({ error: "Authentication failed" }, { status: 500 });
  }
}