import api from "./api";
import type { CallGraphData } from "../types";

export async function getCallGraph(
  repoId: string,
  methodId: string
): Promise<CallGraphData> {
  const { data } = await api.get(
    `/repos/${repoId}/graph/${encodeURIComponent(methodId)}`
  );
  return data;
}
