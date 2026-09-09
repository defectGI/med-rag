import { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";
import { isCited, parseCitations } from "@/lib/api";
import type { EvidenceChunk } from "@/types";

/**
 * Answer text renderer (B6): converts `(Kaynak: X, s. 4)` / `(Source: X, p. 4)`
 * patterns into clickable badges — clicking a badge opens the corresponding
 * evidence file at that part.
 */
export function AnswerText({
  text,
  chunks,
  onOpen,
  className,
}: {
  text: string;
  chunks?: EvidenceChunk[];
  onOpen?: (c: EvidenceChunk) => void;
  className?: string;
}) {
  const rendered = useMemo(() => {
    // (Kaynak: dosya.pdf, s. 12) / (Source: dosya.pdf, p. 12) -> badge
    const re = /\((Kaynak|Source): ([^)]+)\)/g;
    const out: (string | { chip: string })[] = [];
    let last = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(text))) {
      if (m.index > last) out.push(text.slice(last, m.index));
      out.push({ chip: m[0].slice(1, -1) });
      last = m.index + m[0].length;
    }
    if (last < text.length) out.push(text.slice(last));
    return out;
  }, [text]);

  const cited = useMemo(() => {
    if (!chunks?.length) return [];
    const { labels } = parseCitations(text);
    if (labels.length) return chunks.filter((c) => isCited(c, labels));
    return [];
  }, [chunks, text]);

  const findOpen = (chip: string) => {
    if (!onOpen || !cited.length) return;
    const lbl = chip.replace(/^\((?:Kaynak|Source):\s*/, "").split(",")[0].trim();
    const match = cited.find((c) => {
      const t = (lbl || "").toLowerCase();
      return t && `${c.doc_id} ${c.file_name ?? ""}`.toLowerCase().includes(t);
    });
    if (match) onOpen(match);
  };

  return (
    <div className={cn("md-view text-sm", className)}>
      {rendered.map((part, i) =>
        typeof part === "string" ? (
          <MarkdownInline key={i} text={part} />
        ) : (
          <span
            key={i}
            className={cn("citation-chip", cited.length && "cursor-pointer")}
            title={part.chip}
            onClick={() => findOpen(part.chip)}
          >
            {part.chip.split(/[,;]/)[0].split(":").slice(1).join(":").trim().slice(0, 24)}
          </span>
        ),
      )}
    </div>
  );
}

function MarkdownInline({ text }: { text: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        // in-answer table/list styles come from the md-view class
      }}
    >
      {text}
    </ReactMarkdown>
  );
}
