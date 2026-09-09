import { useState } from "react";
import { ExternalLink, FileText, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import type { EvidenceChunk } from "@/types";

function isImage(file: string): boolean {
  return /\.(jpe?g|png)$/i.test(file);
}
function isPdf(file: string): boolean {
  return /\.pdf$/i.test(file);
}
/** Sayfa fragment'ı: PDF tarayıcı yerleşik görüntüleyicide `#page=N` atlar. */
function fileUrl(docId: string, isPdfFile: boolean, page?: string): string {
  const base = api.documentFileUrl(docId);
  if (isPdfFile && page) return `${base}#page=${parseInt(page, 10) || 1}`;
  return base;
}

/**
 * Kanıt görüntüleyici: tıklanan kanıtın SADECE o parçasını ve orijinalde aynı
 * sayfa/bölümünü gösterir (indirmeden). PDF -> gömülü `#page=N`, görsel -> img,
 * diğerleri -> parça metni + orijinali aç.
 */
export function EvidenceViewer({
  chunk,
  onClose,
}: {
  chunk: EvidenceChunk;
  onClose: () => void;
}) {
  const [tab, setTab] = useState<"part" | "original">("part");
  const name = chunk.file_name ?? chunk.doc_id;
  const pdf = isPdf(name);
  const img = isImage(name);
  const page = chunk.page ? String(chunk.page).split("-")[0] : undefined;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" onClick={onClose}>
      <div
        className="flex max-h-[85vh] w-full max-w-3xl flex-col overflow-hidden rounded-lg border bg-background shadow-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 border-b px-4 py-3">
          <FileText className="h-4 w-4 text-primary" />
          <span className="flex-1 truncate text-sm font-medium">{name}</span>
          {chunk.section && <span className="truncate text-xs text-muted-foreground">{chunk.section}</span>}
          {chunk.page && <span className="shrink-0 text-xs text-muted-foreground">s. {chunk.page}</span>}
          <button onClick={onClose} className="ml-1 rounded-md p-1 hover:bg-muted" title="Kapat">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex items-center gap-2 border-b px-4 py-2 text-sm">
          <button
            className={`rounded-md px-3 py-1 ${tab === "part" ? "bg-primary text-primary-foreground" : "hover:bg-muted"}`}
            onClick={() => setTab("part")}
          >
            Bu parça
          </button>
          <button
            className={`rounded-md px-3 py-1 ${tab === "original" ? "bg-primary text-primary-foreground" : "hover:bg-muted"}`}
            onClick={() => setTab("original")}
          >
            Orijinal{page ? ` · s.${page}` : ""}
          </button>
          <a
            href={fileUrl(chunk.doc_id, pdf, page)}
            download={!pdf && !img}
            target="_blank"
            rel="noreferrer"
            className="ml-auto inline-flex"
          >
            <Button variant="ghost" size="sm">
              <ExternalLink className="h-4 w-4" /> Orijinali aç
            </Button>
          </a>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-4">
          {tab === "part" ? (
            <p className="whitespace-pre-wrap text-sm leading-relaxed">{chunk.text}</p>
          ) : pdf ? (
            <iframe
              src={fileUrl(chunk.doc_id, true, page)}
              title={name}
              className="h-[60vh] w-full rounded-md border"
            />
          ) : img ? (
            <img src={api.documentFileUrl(chunk.doc_id)} alt={name} className="mx-auto max-h-[60vh] rounded-md border object-contain" />
          ) : (
            <div className="flex h-[40vh] flex-col items-center justify-center gap-3 text-muted-foreground">
              <p>Bu biçim tarayıcıda gösterilemiyor. Parça metni solda.</p>
              <Button variant="outline" onClick={() => setTab("part")}>
                Parça metnini göster
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
