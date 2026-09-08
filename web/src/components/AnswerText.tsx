import { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";

/**
 * Cevap metni renderer'ı (B6): `(Kaynak: X, s. 4)` / `(Source: X, p. 4)`
 * kalıplarını tıklanabilir rozetlere çevirir.
 */
export function AnswerText({ text, className }: { text: string; className?: string }) {
  const rendered = useMemo(() => {
    // (Kaynak: dosya.pdf, s. 12) / (Source: dosya.pdf, p. 12) -> rozet
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

  return (
    <div className={cn("md-view text-sm", className)}>
      {rendered.map((part, i) =>
        typeof part === "string" ? (
          <MarkdownInline key={i} text={part} />
        ) : (
          <span key={i} className="citation-chip" title={part.chip}>
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
        // cevap içi tablo/liste stilleri md-view sınıfından gelir
      }}
    >
      {text}
    </ReactMarkdown>
  );
}
