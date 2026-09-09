import { useCallback, useEffect, useState } from "react";
import { Toaster } from "sonner";
import { Sidebar, type View } from "@/components/Sidebar";
import { ChatPage } from "@/pages/ChatPage";
import { LibraryPage } from "@/pages/LibraryPage";
import { LoginPage } from "@/pages/LoginPage";
import { NotesPage } from "@/pages/NotesPage";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { DocumentItem } from "@/types";

export default function App() {
  const [authed, setAuthed] = useState<boolean | null>(null);
  const [view, setView] = useState<View>("library");
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [notes, setNotes] = useState<DocumentItem[]>([]);

  const refresh = useCallback(async () => {
    try {
      const [docs, nts] = await Promise.all([api.documents(), api.notes()]);
      setDocuments(docs);
      setNotes(nts);
    } catch {
      // 401 -> oturum düşmüş
      setAuthed(false);
    }
  }, []);

  useEffect(() => {
    api.session()
      .then((s) => setAuthed(s.authenticated || !s.required))
      .catch(() => setAuthed(false));
  }, []);

  useEffect(() => {
    if (authed !== true) return;
    void refresh();
    const es = api.openDocumentStream((docs) => {
      setDocuments(docs);
      setNotes(docs.filter((d) => d.doc_type === "NOTE"));
    });
    return () => es.close();
  }, [authed, refresh]);

  if (authed === null) {
    return <div className="flex min-h-screen items-center justify-center text-muted-foreground">…</div>;
  }
  if (!authed) {
    return (
      <>
        <LoginPage onLogin={() => setAuthed(true)} />
        <Toaster richColors position="top-center" />
      </>
    );
  }

  const pendingCount = documents.filter((d) =>
    ["queued", "parsing", "chunking", "vectorizing"].includes(d.status.state),
  ).length;

  return (
    <div className="flex h-screen">
      <Sidebar
        view={view}
        onView={setView}
        pendingCount={pendingCount}
        onLogout={async () => {
          await api.logout();
          setAuthed(false);
        }}
      />
      <main className="flex-1 overflow-y-auto p-5">
        {/* Üç görünüm de her zaman mount kalır; etkin olmayanlar CSS'le
            gizlenir. Koşullu render (&&) unmount yapıp sohbet mesajlarını ve
            yazılmakta olan not taslağını kaybettiriyordu (sekme geçişi). */}
        <div className={cn("h-full", view !== "library" && "hidden")}>
          <LibraryPage
            documents={documents}
            onUploaded={() => void refresh()}
            onDelete={(docId) => void api.deleteDocument(docId).then(refresh)}
          />
        </div>
        <div className={cn("h-full", view !== "chat" && "hidden")}>
          <ChatPage />
        </div>
        <div className={cn("h-full", view !== "notes" && "hidden")}>
          <NotesPage
            notes={notes}
            onChanged={() => void refresh()}
            onDelete={(docId) => void api.deleteDocument(docId).then(refresh)}
          />
        </div>
      </main>
      <Toaster richColors position="top-center" />
    </div>
  );
}
