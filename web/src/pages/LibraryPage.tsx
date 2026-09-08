import { useCallback, useRef, useState } from "react";
import { Eye, FileDown, Loader2, Search, Trash2, Upload } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { AnswerText } from "@/components/AnswerText";
import { StatusBadge } from "@/components/StatusBadge";
import { api, formatBytes, formatDate } from "@/lib/api";
import type { DocumentItem } from "@/types";

const ACCEPT = ".pdf,.docx,.pptx,.xlsx,.html,.htm,.md,.markdown";

export function LibraryPage({
  documents,
  onUploaded,
  onDelete,
}: {
  documents: DocumentItem[];
  onUploaded: () => void;
  onDelete: (docId: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [viewer, setViewer] = useState<{ file: string; markdown: string } | null>(null);
  const [pendingDelete, setPendingDelete] = useState<DocumentItem | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const upload = useCallback(
    async (files: FileList | File[] | null) => {
      if (!files || files.length === 0) return;
      setBusy(true);
      try {
        const res = await api.upload(Array.from(files));
        for (const r of res.rejected) toast.error(`${r.file_name}: ${r.reason}`);
        if (res.accepted.length > 0) {
          toast.success(`${res.accepted.length} dosya kuyruğa alındı`);
          onUploaded();
        }
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "yükleme başarısız");
      } finally {
        setBusy(false);
      }
    },
    [onUploaded],
  );

  const openViewer = async (doc: DocumentItem) => {
    try {
      const content = await api.documentContent(doc.doc_id);
      setViewer({ file: content.file_name, markdown: content.markdown });
    } catch {
      toast.error("Ayrıştırılmış içerik henüz yok");
    }
  };

  const filtered = documents.filter((d) =>
    (d.note_title ?? d.file_name).toLowerCase().includes(query.toLowerCase()),
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <h1 className="text-xl font-semibold">Kütüphane</h1>
        <span className="text-sm text-muted-foreground">
          {documents.length} belge
        </span>
        <div className="ml-auto flex items-center gap-2">
          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              className="w-56 pl-8"
              placeholder="Belge ara…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
          <Button disabled={busy} onClick={() => fileInput.current?.click()}>
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
            Yükle
          </Button>
          <input
            ref={fileInput}
            type="file"
            multiple
            accept={ACCEPT}
            className="hidden"
            onChange={(e) => {
              void upload(e.target.files);
              e.target.value = "";
            }}
          />
        </div>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          void upload(e.dataTransfer.files);
        }}
        className={`rounded-lg border-2 border-dashed p-4 text-center text-sm text-muted-foreground transition-colors ${
          dragOver ? "border-primary bg-primary/5" : "border-border"
        }`}
      >
        Dosyaları buraya sürükleyip bırakın (PDF, DOCX, PPTX, XLSX, HTML, MD)
      </div>

      <div className="overflow-hidden rounded-lg border bg-card">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50 text-left text-muted-foreground">
              <th className="px-4 py-2.5 font-medium">Belge</th>
              <th className="px-4 py-2.5 font-medium">Tür</th>
              <th className="px-4 py-2.5 font-medium">Boyut</th>
              <th className="px-4 py-2.5 font-medium">Tarih</th>
              <th className="px-4 py-2.5 font-medium">Durum</th>
              <th className="px-4 py-2.5" />
            </tr>
          </thead>
          <tbody>
            {filtered.map((doc) => (
              <tr key={doc.doc_id} className="border-b last:border-0 hover:bg-muted/30">
                <td className="max-w-72 truncate px-4 py-2.5 font-medium">
                  {doc.note_title ?? doc.file_name}
                </td>
                <td className="px-4 py-2.5">
                  <Badge variant="outline">{doc.doc_type ?? "BELGE"}</Badge>
                </td>
                <td className="px-4 py-2.5 text-muted-foreground">{formatBytes(doc.size_bytes)}</td>
                <td className="px-4 py-2.5 text-muted-foreground">{formatDate(doc.uploaded_at)}</td>
                <td className="px-4 py-2.5">
                  <StatusBadge state={doc.status.state} />
                  {doc.status.detail && doc.status.state === "error" && (
                    <p className="mt-1 max-w-64 truncate text-xs text-destructive" title={doc.status.detail}>
                      {doc.status.detail}
                    </p>
                  )}
                </td>
                <td className="px-4 py-2.5">
                  <div className="flex justify-end gap-1">
                    <Button variant="ghost" size="icon" title="İçeriği gör" onClick={() => void openViewer(doc)}>
                      <Eye className="h-4 w-4" />
                    </Button>
                    <a href={api.documentFileUrl(doc.doc_id)} download>
                      <Button variant="ghost" size="icon" title="Orijinali indir">
                        <FileDown className="h-4 w-4" />
                      </Button>
                    </a>
                    <Button
                      variant="ghost"
                      size="icon"
                      title="Sil"
                      className="text-destructive hover:text-destructive"
                      onClick={() => setPendingDelete(doc)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={6} className="px-4 py-10 text-center text-muted-foreground">
                  Henüz belge yok — yukarıdan yükleyin.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Belge görüntüleyici (B7): düzenlenmiş markdown */}
      <Dialog open={viewer !== null} onOpenChange={(o) => !o && setViewer(null)}>
        <DialogContent className="max-h-[85vh] max-w-3xl overflow-y-auto">
          <DialogTitle className="flex items-center gap-2">
            <Eye className="h-4 w-4 text-primary" /> {viewer?.file}
          </DialogTitle>
          <div className="md-view">
            <AnswerText text={viewer?.markdown ?? ""} />
          </div>
        </DialogContent>
      </Dialog>

      {/* Silme onayı */}
      <Dialog open={pendingDelete !== null} onOpenChange={(o) => !o && setPendingDelete(null)}>
        <DialogContent className="max-w-md">
          <DialogTitle>Belgeyi sil</DialogTitle>
          <DialogDescription>
            “{pendingDelete?.note_title ?? pendingDelete?.file_name}” ve bu belgeden türeyen tüm
            veriler (chunk'lar, vektörler) kalıcı olarak silinecek.
          </DialogDescription>
          <div className="mt-4 flex justify-end gap-2">
            <DialogClose asChild>
              <Button variant="outline">Vazgeç</Button>
            </DialogClose>
            <Button
              variant="destructive"
              onClick={() => {
                if (pendingDelete) onDelete(pendingDelete.doc_id);
                setPendingDelete(null);
              }}
            >
              Kalıcı sil
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
