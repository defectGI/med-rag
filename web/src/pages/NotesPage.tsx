import { useEffect, useState } from "react";
import { Loader2, Plus, Save, StickyNote, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { StatusBadge } from "@/components/StatusBadge";
import { Textarea } from "@/components/ui/textarea";
import { api } from "@/lib/api";
import type { NoteItem } from "@/types";

export function NotesPage({
  notes,
  onChanged,
  onDelete,
}: {
  notes: NoteItem[];
  onChanged: () => void;
  onDelete: (docId: string) => void;
}) {
  const [selected, setSelected] = useState<NoteItem | null>(null);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (selected) {
      const fresh = notes.find((n) => n.doc_id === selected.doc_id);
      if (fresh) setSelected(fresh);
    }
  }, [notes, selected]);

  const open = (note: NoteItem) => {
    setSelected(note);
    setTitle(note.note_title ?? note.file_name);
    setContent("");
    void api
      .documentContent(note.doc_id)
      .then((c) => setContent(c.markdown))
      .catch(() => setContent(""));
  };

  const createNew = async () => {
    setBusy(true);
    try {
      const note = await api.createNote("Yeni not", "");
      onChanged();
      open(note);
      toast.success("Not oluşturuldu");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "not oluşturulamadı");
    } finally {
      setBusy(false);
    }
  };

  const save = async () => {
    if (!selected) return;
    setBusy(true);
    try {
      await api.updateNote(selected.doc_id, { title, content });
      onChanged();
      toast.success("Not kaydedildi — sohbette kullanılabilir hale geliyor");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "kaydedilemedi");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-full gap-4">
      <div className="flex w-72 flex-col gap-2">
        <div className="flex items-center gap-2">
          <h1 className="text-xl font-semibold">Notlar</h1>
          <Button size="sm" variant="outline" className="ml-auto" disabled={busy} onClick={() => void createNew()}>
            <Plus className="h-4 w-4" /> Yeni
          </Button>
        </div>
        <div className="flex-1 space-y-1 overflow-y-auto rounded-lg border bg-card p-2">
          {notes.map((note) => (
            <button
              key={note.doc_id}
              onClick={() => open(note)}
              className={`flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors ${
                selected?.doc_id === note.doc_id ? "bg-primary/15 text-primary" : "hover:bg-muted"
              }`}
            >
              <StickyNote className="h-4 w-4 shrink-0" />
              <span className="flex-1 truncate">{note.note_title ?? note.file_name}</span>
              <StatusBadge state={note.status.state} />
            </button>
          ))}
          {notes.length === 0 && (
            <p className="px-3 py-8 text-center text-sm text-muted-foreground">
              Henüz not yok — "Yeni" ile başlayın. Notlar sohbete kaynak olur.
            </p>
          )}
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-2">
        {selected ? (
          <>
            <div className="flex items-center gap-2">
              <Input value={title} onChange={(e) => setTitle(e.target.value)} className="max-w-md" />
              <Button size="sm" disabled={busy} onClick={() => void save()}>
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
                Kaydet
              </Button>
              <Button
                size="sm"
                variant="ghost"
                className="ml-auto text-destructive hover:text-destructive"
                onClick={() => {
                  onDelete(selected.doc_id);
                  setSelected(null);
                }}
              >
                <Trash2 className="h-4 w-4" /> Sil
              </Button>
            </div>
            <Textarea
              className="flex-1 resize-none font-mono text-[13px]"
              placeholder="Not içeriği… (markdown destekler)"
              value={content}
              onChange={(e) => setContent(e.target.value)}
            />
            <p className="text-xs text-muted-foreground">
              Kaydettiğinizde not ayrıştırılıp vektöre yazılır; chatbot
              sorularınızda bu notu kaynak olarak kullanabilir.
            </p>
          </>
        ) : (
          <div className="flex flex-1 items-center justify-center rounded-lg border border-dashed text-muted-foreground">
            Soldan bir not seçin ya da yeni not oluşturun.
          </div>
        )}
      </div>
    </div>
  );
}
