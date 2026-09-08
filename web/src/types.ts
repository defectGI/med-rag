export type DocState =
  | "queued"
  | "parsing"
  | "chunking"
  | "vectorizing"
  | "ready"
  | "error";

export interface DocStatus {
  state: DocState;
  detail: string | null;
  updated_at: string | null;
}

export interface DocumentItem {
  doc_id: string;
  file_name: string;
  doc_type: string | null;
  rel_path: string | null;
  size_bytes: number | null;
  uploaded_at: string | null;
  note_title?: string | null;
  status: DocStatus;
}

export interface NoteItem extends DocumentItem {}

export interface EvidenceChunk {
  doc_id: string;
  id: string;
  file_name: string | null;
  section: string | null;
  page: string | null;
  text: string;
  kind?: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  text: string;
  chunks?: EvidenceChunk[];
  error?: boolean;
  streaming?: boolean;
}

export interface StepEvent {
  step: string;
  data: unknown;
}

export const STATE_LABELS: Record<DocState, string> = {
  queued: "Kuyrukta",
  parsing: "İşleniyor",
  chunking: "Chunk'lanıyor",
  vectorizing: "Gömülüyor",
  ready: "Hazır",
  error: "Hata",
};
