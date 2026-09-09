import type {
  ChatMessage,
  DocumentItem,
  EvidenceChunk,
  NoteItem,
  StepEvent,
} from "@/types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function json_or_error<T>(resp: Response): Promise<T> {
  if (resp.status === 401) throw new ApiError(401, "giriş gerekli");
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new ApiError(resp.status, (body as { error?: string }).error ?? resp.statusText);
  }
  return resp.json() as Promise<T>;
}

export const api = {
  async session(): Promise<{ authenticated: boolean; required: boolean }> {
    return json_or_error(await fetch(`${BASE}/api/auth/session`));
  },
  async login(password: string): Promise<void> {
    const resp = await fetch(`${BASE}/api/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    });
    if (resp.status === 401) throw new ApiError(401, "şifre hatalı");
    if (!resp.ok) throw new ApiError(resp.status, "giriş başarısız");
  },
  async logout(): Promise<void> {
    await fetch(`${BASE}/api/auth/logout`, { method: "POST" });
  },

  async documents(): Promise<DocumentItem[]> {
    const data = await json_or_error<{ documents: DocumentItem[] }>(
      await fetch(`${BASE}/api/library/documents`),
    );
    return data.documents;
  },
  async upload(files: File[]): Promise<{ accepted: DocumentItem[]; rejected: { file_name: string; reason: string }[] }> {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return json_or_error(
      await fetch(`${BASE}/api/library/documents`, { method: "POST", body: form }),
    );
  },
  async deleteDocument(docId: string): Promise<void> {
    const resp = await fetch(`${BASE}/api/library/documents/${docId}`, { method: "DELETE" });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new ApiError(resp.status, (body as { error?: string }).error ?? "silme başarısız");
    }
  },
  async documentContent(docId: string): Promise<{ file_name: string; markdown: string }> {
    return json_or_error(await fetch(`${BASE}/api/library/documents/${docId}/content`));
  },
  documentFileUrl(docId: string): string {
    return `${BASE}/api/library/documents/${docId}/file`;
  },

  async notes(): Promise<NoteItem[]> {
    const data = await json_or_error<{ notes: NoteItem[] }>(
      await fetch(`${BASE}/api/library/notes`),
    );
    return data.notes;
  },
  async createNote(title: string, content: string): Promise<NoteItem> {
    const data = await json_or_error<{ note: NoteItem }>(
      await fetch(`${BASE}/api/library/notes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, content }),
      }),
    );
    return data.note;
  },
  async updateNote(noteId: string, patch: { title?: string; content?: string }): Promise<void> {
    const resp = await fetch(`${BASE}/api/library/notes/${noteId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new ApiError(resp.status, (body as { error?: string }).error ?? "kaydedilemedi");
    }
  },
  async formatNote(noteId: string): Promise<string> {
    const data = await json_or_error<{ ok: boolean; content: string }>(
      await fetch(`${BASE}/api/library/notes/${noteId}/format`, { method: "POST" }),
    );
    return data.content;
  },

  async history(): Promise<ChatMessage[]> {
    const data = await json_or_error<{
      messages: {
        role: "user" | "assistant";
        text: string;
        chunks?: unknown[];
        error?: boolean;
      }[];
    }>(await fetch(`${BASE}/api/chat/history`));
    return data.messages.map((m) => ({
      role: m.role,
      text: m.text,
      error: m.error,
      chunks: normalizeChunks(m.chunks ?? []),
    }));
  },
  async reset(): Promise<void> {
    const resp = await fetch(`${BASE}/api/reset`, { method: "POST" });
    if (!resp.ok) throw new ApiError(resp.status, "yeni sohbet başlatılamadı");
  },

  /** Sohbet akışı: SSE olaylarını geri çağırmalara dağıtır. */
  streamChat(
    message: string,
    handlers: {
      onStep?: (ev: StepEvent) => void;
      onDone?: (reply: string, chunks: EvidenceChunk[]) => void;
      onError?: (message: string) => void;
    },
    signal?: AbortSignal,
  ): Promise<void> {
    return fetch(`${BASE}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
      signal,
    }).then(async (resp) => {
      if (resp.status === 401) throw new ApiError(401, "giriş gerekli");
      if (!resp.ok || !resp.body) throw new ApiError(resp.status, "akış başlatılamadı");
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";
        for (const part of parts) {
          let event = "message";
          let data = "";
          for (const line of part.split("\n")) {
            if (line.startsWith("event: ")) event = line.slice(7).trim();
            else if (line.startsWith("data: ")) data += line.slice(6);
          }
          if (!data) continue;
          const payload = JSON.parse(data);
          if (event === "step") handlers.onStep?.(payload as StepEvent);
          else if (event === "done") {
            const p = payload as { reply: string; chunks: unknown[] };
            handlers.onDone?.(p.reply ?? "", normalizeChunks(p.chunks ?? []));
          } else if (event === "error") {
            handlers.onError?.((payload as { error: string }).error);
          }
        }
      }
    });
  },

  /** Kütüphane durum akışı: her anlık görüntüde callback tetiklenir. */
  openDocumentStream(onDocuments: (docs: DocumentItem[]) => void): EventSource {
    const es = new EventSource(`${BASE}/api/library/events`);
    es.addEventListener("documents", (ev) => {
      const data = JSON.parse((ev as MessageEvent).data as string);
      onDocuments(data.documents as DocumentItem[]);
    });
    return es;
  },
};

/** Arka uç `snippet`/`source_kind` alanlarını SPA'nın `text`/`kind` şekline
 * çevirir -- hem canlı akış hem geçmiş aynı şekli kullanır. */
function normalizeChunks(chunks: unknown[]): EvidenceChunk[] {
  return (chunks ?? []).map((c) => {
    const raw = (c ?? {}) as Record<string, unknown>;
    return {
      doc_id: (raw.doc_id as string) ?? "",
      id: (raw.id as string) ?? "",
      file_name: (raw.file_name as string | null) ?? null,
      section: (raw.section as string | null) ?? null,
      page: (raw.page as string | null) ?? null,
      text: (raw.snippet as string) ?? (raw.text as string) ?? "",
      kind: (raw.source_kind as string | undefined) ?? (raw.kind as string | undefined),
    };
  });
}

export function formatBytes(n: number | null | undefined): string {
  if (n == null) return "–";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "–";
  return new Date(iso).toLocaleString("tr-TR", {
    day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

export type { ChatMessage };
