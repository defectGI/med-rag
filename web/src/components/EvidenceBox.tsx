import { ChevronDown, FileText, ScrollText } from "lucide-react";
import { useState } from "react";
import type { EvidenceChunk } from "@/types";
import { cn } from "@/lib/utils";

/**
 * Kanıt kutusu (B6): cevabın dayandığı kaynak chunk'lar -- numaralı liste,
 * belge adı + sayfa + bölüm; satır içi [n] rozetleriyle eşleşir.
 */
export function EvidenceBox({ chunks }: { chunks: EvidenceChunk[] }) {
  const [open, setOpen] = useState(false);
  if (!chunks?.length) return null;
  return (
    <div className="mt-2 rounded-md border bg-muted/40">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground"
      >
        <ScrollText className="h-3.5 w-3.5" />
        Kanıtlar ({chunks.length})
        <ChevronDown className={cn("ml-auto h-3.5 w-3.5 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <ol className="space-y-2 border-t px-3 py-2">
          {chunks.map((c, i) => (
            <li key={c.id ?? i} className="text-xs">
              <span className="citation-chip mr-1.5">[{i + 1}]</span>
              <span className="font-medium inline-flex items-center gap-1">
                <FileText className="inline h-3 w-3" />
                {c.file_name ?? c.doc_id}
              </span>
              {(c.page || c.section) && (
                <span className="text-muted-foreground">
                  {" · "}
                  {c.section && <>{c.section} · </>}
                  {c.page && <>s. {c.page}</>}
                </span>
              )}
              <p className="mt-1 line-clamp-3 text-muted-foreground">{c.text}</p>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
