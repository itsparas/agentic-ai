import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata = {
  title: "LLM Study Guide — Senior Agentic Engineer Prep",
  description:
    "Interactive study guide covering LLM internals, ML foundations, agent patterns, RAG, production, and safety.",
};

export default function RootLayout({ children }) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="flex h-full bg-white text-gray-900 dark:bg-gray-950 dark:text-gray-100">
        <Sidebar />
        <main className="flex-1 overflow-y-auto p-10">{children}</main>
      </body>
    </html>
  );
}
