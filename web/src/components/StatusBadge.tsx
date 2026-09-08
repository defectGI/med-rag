import { Badge } from "@/components/ui/badge";
import { STATE_LABELS, type DocState } from "@/types";

const VARIANT: Record<DocState, "secondary" | "warn" | "success" | "destructive"> = {
  queued: "secondary",
  parsing: "warn",
  chunking: "warn",
  vectorizing: "warn",
  ready: "success",
  error: "destructive",
};

export function StatusBadge({ state }: { state: DocState }) {
  return <Badge variant={VARIANT[state]}>{STATE_LABELS[state]}</Badge>;
}
