import { useEffect, useRef } from "react";
import { AlertTriangle, FileText, MessageSquareText, StickyNote } from "lucide-react";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export type View = "library" | "chat" | "notes";

const ITEMS: { key: View; label: string; icon: typeof FileText }[] = [
  { key: "library", label: "Kütüphane", icon: FileText },
  { key: "chat", label: "Sohbet", icon: MessageSquareText },
  { key: "notes", label: "Notlar", icon: StickyNote },
];

export function Sidebar({
  view,
  onView,
  pendingCount,
  onLogout,
}: {
  view: View;
  onView: (v: View) => void;
  pendingCount: number;
  onLogout: () => void;
}) {
  const firstRun = useRef(true);
  useEffect(() => {
    firstRun.current = false;
  }, []);

  return (
    <aside className="flex h-full w-56 flex-col border-r bg-card">
      <div className="flex items-center gap-2 px-4 py-4">
        <span className="inline-flex h-8 w-8 items-center justify-center rounded-md bg-primary text-primary-foreground font-bold">
          M
        </span>
        <div>
          <div className="font-semibold leading-none">med-rag</div>
          <div className="text-xs text-muted-foreground">tıbbi belge asistanı</div>
        </div>
      </div>
      <nav className="flex-1 space-y-1 px-2">
        {ITEMS.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => onView(key)}
            className={cn(
              "flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors",
              view === key ? "bg-primary/15 text-primary font-medium" : "hover:bg-muted",
            )}
          >
            <Icon className="h-4 w-4" />
            {label}
            {key === "library" && pendingCount > 0 && (
              <span className="ml-auto inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-warn/20 px-1.5 text-[11px] font-medium text-warn">
                {pendingCount}
              </span>
            )}
          </button>
        ))}
      </nav>
      <div className="flex items-center justify-between px-3 py-3">
        <ThemeToggle />
        <Button variant="ghost" size="sm" onClick={onLogout}>
          Çıkış
        </Button>
      </div>
      {firstRun.current && (
        <div className="hidden">
          <AlertTriangle />
        </div>
      )}
    </aside>
  );
}
