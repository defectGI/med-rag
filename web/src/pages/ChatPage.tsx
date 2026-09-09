import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, Plus, Send } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { AnswerText } from "@/components/AnswerText";
import { EvidenceBox } from "@/components/EvidenceBox";
import { EvidenceViewer } from "@/components/EvidenceViewer";
import { api } from "@/lib/api";
import type { ChatMessage, EvidenceChunk, StepEvent } from "@/types";

export function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [currentStep, setCurrentStep] = useState<string | null>(null);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [viewChunk, setViewChunk] = useState<EvidenceChunk | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api
      .history()
      .then((msgs) => setMessages(msgs))
      .catch(() => {
        // 401 -> session dropped; App already switches to the login screen.
      })
      .finally(() => setHistoryLoaded(true));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentStep]);

  const newChat = useCallback(async () => {
    setStreaming(false);
    setMessages([]);
    setInput("");
    setCurrentStep(null);
    try {
      await api.reset();
      toast.success("Yeni sohbet başlatıldı");
    } catch {
      // Even if the server couldn't reset, the UI has been cleared; on the next
      // message the session just continues with a new cookie.
    }
  }, []);

  const send = async () => {
    const message = input.trim();
    if (!message || streaming) return;
    setInput("");
    setStreaming(true);
    setCurrentStep(null);
    setMessages((m) => [...m, { role: "user", text: message }, { role: "assistant", text: "", streaming: true }]);

    const onStep = (ev: StepEvent) => {
      if (ev.step === "preamble") return;
      setCurrentStep(stepLabel(ev.step, ev.data));
    };
    const onDone = (reply: string, chunks: EvidenceChunk[]) => {
      setMessages((m) => {
        const copy = [...m];
        copy[copy.length - 1] = { role: "assistant", text: reply, chunks };
        return copy;
      });
      setStreaming(false);
      setCurrentStep(null);
    };
    const onError = (error: string) => {
      setMessages((m) => {
        const copy = [...m];
        copy[copy.length - 1] = { role: "assistant", text: error, error: true };
        return copy;
      });
      setStreaming(false);
      setCurrentStep(null);
    };

    try {
      await api.streamChat(message, { onStep, onDone, onError });
    } catch (err) {
      onError(err instanceof Error ? err.message : "bağlantı hatası");
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between pb-3">
        <h1 className="text-xl font-semibold">Sohbet</h1>
        <Button variant="outline" size="sm" onClick={() => void newChat()}>
          <Plus className="h-4 w-4" />
          Yeni sohbet
        </Button>
      </div>
      <div className="flex-1 space-y-4 overflow-y-auto rounded-lg border bg-card p-4">
        {!historyLoaded ? (
          <div className="flex h-full items-center justify-center text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
          </div>
        ) : messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-muted-foreground">
            <p className="text-lg font-medium">Kütüphanenize sorun</p>
            <p className="max-w-md text-center text-sm">
              Yüklediğiniz belgelerden, kanıt gösterilerek cevaplanır — cevap
              bulunamazsa bunu dürüstçe söyler.
            </p>
          </div>
        ) : null}
        {messages.map((msg, i) =>
          msg.role === "user" ? (
            <div key={i} className="flex justify-end">
              <div className="max-w-[80%] whitespace-pre-wrap rounded-lg bg-primary px-3.5 py-2 text-sm text-primary-foreground">
                {msg.text}
              </div>
            </div>
          ) : (
            <div key={i} className="flex justify-start">
              <div
                className={`max-w-[85%] rounded-lg border px-3.5 py-2.5 text-sm ${
                  msg.error ? "border-destructive/40 bg-destructive/10 text-destructive" : "bg-background"
                }`}
              >
                {msg.streaming && !msg.text ? (
                  <span className="flex items-center gap-2 text-muted-foreground">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    {currentStep ?? "kaynaklara bakılıyor…"}
                  </span>
                ) : (
                  <>
                    <AnswerText text={msg.text} chunks={msg.chunks ?? []} onOpen={setViewChunk} />
                    <EvidenceBox chunks={msg.chunks ?? []} answerText={msg.text} onOpen={setViewChunk} />
                  </>
                )}
              </div>
            </div>
          ),
        )}
        <div ref={bottomRef} />
      </div>
      <form
        className="mt-3 flex items-end gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
      >
        <Textarea
          rows={1}
          placeholder="Sorunuzu yazın… (ör. 'Venlafaksin başlangıç dozu ne?')"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
          className="max-h-40"
        />
        <Button type="submit" size="icon" disabled={streaming || !input.trim()} title="Gönder">
          {streaming ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
        </Button>
      </form>
      {viewChunk && <EvidenceViewer chunk={viewChunk} onClose={() => setViewChunk(null)} />}
    </div>
  );
}

function stepLabel(step: string, data: unknown): string {
  const detay =
    typeof data === "object" && data !== null && "flow_name" in (data as Record<string, unknown>)
      ? ` (${(data as { flow_name: string }).flow_name})`
      : "";
  const labels: Record<string, string> = {
    intent: "niyet anlaşılıyor…",
    routed: "yol seçildi…",
    routed_default_fallback: "yol seçildi…",
    chunks: "kaynaklar bulundu…",
    sql: "sorgu hazırlanıyor…",
    preamble: "",
  };
  return (labels[step] ?? step) + detay;
}
