import { ChevronDown, FileText, ScrollText } from "lucide-react";
import { useMemo, useState } from "react";
import type { EvidenceChunk } from "@/types";
import { cn } from "@/lib/utils";
import { isCited, parseCitations } from "@/lib/api";

/**
 * Kanıt kutusu (B6): cevabın dayandığı kaynak chunk'lar. Cevap metnindeki
 * citation'lar çözümlenip YALNIZCA cite edilen kanıtlar gösterilir (top-N'in
 * tamamı değil). Her satır tıklanabilir -> ilgili dosyanın o parçası açılır.
 */
export function EvidenceBox({
  chunks,
  answerText,
  onOpen,
}: {
  chunks: EvidenceChunk[];
  answerText?: string;
  onOpen: (chunk: EvidenceChunk) => void;
}) {
  const [open, setOpen] = useState(true);

  const selected = useMemo(() => {
    if (!chunks?.length) return [];
    const { labels, indices } = parseCitations(answerText ?? "");
    // Etiket eşleşmesi çalışırsa tam o kanıtları göster.
    if (labels.length) {
      const cited = chunks.filter((c) => isCited(c, labels));
      if (cited.length) return cited;
    }
    if (indices.length) {
      const byIndex = indices
        .map((n) => chunks[n - 1])
        .filter((c): c is EvidenceChunk => Boolean(c));
      if (byIndex.length) return byIndex;
    }
    // Citation'lar çözülemezse: cevabın cite ettiği KAYNAK SAYISI kadar en
    // alakalı chunk'ı göster (asso model başlık kullanıp eşleşmezse). Top-N'in
    // tamamını (10) dökmek yerine yalnız cevabın dayandığı kadarı.
    const k = labels.length || indices.length;
    return chunks.slice(0, Math.max(1, k));
  }, [chunks, answerText]);

  if (!selected.length) return null;
  return (
    <div className="mt-2 rounded-md border bg-muted/40">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground"
      >
        <ScrollText className="h-3.5 w-3.5" />
        Kanıtlar ({selected.length})
        <ChevronDown className={cn("ml-auto h-3.5 w-3.5 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <ol className="space-y-2 border-t px-3 py-2">
          {selected.map((c, i) => (
            <li key={c.id ?? i} className="text-xs">
              <button
                onClick={() => onOpen(c)}
                className="group flex w-full items-start gap-1.5 rounded-md p-1.5 text-left transition-colors hover:bg-background"
                title="Kanıtı aç — orijinalin o parçasına gider"
              >
                <span className="citation-chip mt-0.5 shrink-0">[{i + 1}]</span>
                <span className="min-w-0 flex-1">
                  <span className="font-medium inline-flex items-center gap-1">
                    <FileText className="inline h-3 w-3" />
                    <span className="truncate">{c.file_name ?? c.doc_id}</span>
                  </span>
                  {(c.page || c.section) && (
                    <span className="text-muted-foreground">
                      {" · "}
                      {c.section && <>{c.section} · </>}
                      {c.page && <>s. {c.page}</>}
                    </span>
                  )}
                  <span className="mt-1 block line-clamp-3 text-muted-foreground group-hover:line-clamp-none">
                    {c.text}
                  </span>
                </span>
              </button>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
