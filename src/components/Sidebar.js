"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { topics, categories } from "@/data/topics";

export default function Sidebar() {
  const pathname = usePathname();

  const grouped = categories.map((cat) => ({
    name: cat,
    items: topics.filter((t) => t.category === cat),
  }));

  return (
    <aside className="w-72 shrink-0 border-r border-gray-200 bg-gray-50 p-6 overflow-y-auto dark:border-gray-800 dark:bg-gray-950">
      <Link href="/" className="block mb-6">
        <h2 className="text-lg font-bold text-gray-900 dark:text-gray-100">
          LLM Study Guide
        </h2>
      </Link>
      <nav className="flex flex-col gap-5">
        {grouped.map((group) => (
          <div key={group.name}>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-500">
              {group.name}
            </h3>
            <div className="flex flex-col gap-1">
              {group.items.map((t) => {
                const isActive = pathname === `/topics/${t.id}`;
                return (
                  <Link
                    key={t.id}
                    href={`/topics/${t.id}`}
                    className={`rounded-md px-3 py-1.5 text-sm transition-colors ${
                      isActive
                        ? "bg-blue-100 text-blue-700 font-medium dark:bg-blue-900/30 dark:text-blue-400"
                        : "text-gray-600 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-800 dark:hover:text-gray-200"
                    }`}
                  >
                    {t.title}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>
    </aside>
  );
}
