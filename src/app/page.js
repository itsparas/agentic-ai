import { topics, categories } from "@/data/topics";
import Link from "next/link";

export default function Home() {
  const grouped = categories.map((cat) => ({
    name: cat,
    items: topics.filter((t) => t.category === cat),
  }));

  return (
    <div className="max-w-3xl">
      <h1 className="mb-2 text-3xl font-bold">
        Senior Agentic Engineer — Study Guide
      </h1>
      <p className="mb-10 text-gray-500 dark:text-gray-400">
        LLM internals, ML foundations, agent patterns, RAG, production
        reliability, and safety. Click any topic to study.
      </p>

      <div className="flex flex-col gap-10">
        {grouped.map((group) => (
          <section key={group.name}>
            <h2 className="mb-4 text-lg font-semibold text-gray-700 dark:text-gray-300">
              {group.name}
            </h2>
            <div className="grid gap-3 sm:grid-cols-2">
              {group.items.map((t) => (
                <Link
                  key={t.id}
                  href={`/topics/${t.id}`}
                  className="rounded-lg border border-gray-200 p-4 transition-colors hover:border-blue-400 hover:bg-blue-50/50 dark:border-gray-800 dark:hover:border-blue-600 dark:hover:bg-blue-900/10"
                >
                  <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                    {t.title}
                  </h3>
                  <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                    {t.summary}
                  </p>
                </Link>
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
