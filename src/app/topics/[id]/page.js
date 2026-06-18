import { topics } from "@/data/topics";
import { notFound } from "next/navigation";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function generateStaticParams() {
  return topics.map((t) => ({ id: t.id }));
}

export async function generateMetadata({ params }) {
  const { id } = await params;
  const topic = topics.find((t) => t.id === id);
  if (!topic) return {};
  return {
    title: `${topic.title} — LLM Study Guide`,
    description: topic.summary,
  };
}

export default async function TopicPage({ params }) {
  const { id } = await params;
  const index = topics.findIndex((t) => t.id === id);
  if (index === -1) notFound();

  const topic = topics[index];
  const prev = topics[index - 1];
  const next = topics[index + 1];

  return (
    <article className="max-w-2xl">
      <div className="mb-4">
        <span className="text-xs font-semibold uppercase tracking-wider text-blue-600 dark:text-blue-400">
          {topic.category}
        </span>
      </div>
      <h1 className="mb-6 text-3xl font-bold">{topic.title}</h1>

      <div className="prose prose-gray dark:prose-invert max-w-none">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {topic.body}
        </ReactMarkdown>
      </div>

      <nav className="mt-12 flex justify-between border-t border-gray-200 pt-6 dark:border-gray-800">
        {prev ? (
          <Link
            href={`/topics/${prev.id}`}
            className="text-sm text-blue-600 hover:underline dark:text-blue-400"
          >
            &larr; {prev.title}
          </Link>
        ) : (
          <span />
        )}
        {next ? (
          <Link
            href={`/topics/${next.id}`}
            className="text-sm text-blue-600 hover:underline dark:text-blue-400"
          >
            {next.title} &rarr;
          </Link>
        ) : (
          <Link
            href="/"
            className="text-sm text-blue-600 hover:underline dark:text-blue-400"
          >
            Back to overview &rarr;
          </Link>
        )}
      </nav>
    </article>
  );
}
