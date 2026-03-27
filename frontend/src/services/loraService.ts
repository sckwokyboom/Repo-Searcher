import api from "./api";
import type { LoRAStatus, LoRAAdapterInfo } from "../types";

export async function getAvailableAdapters(): Promise<LoRAAdapterInfo[]> {
  const { data } = await api.get("/lora/adapters");
  return data;
}

export async function selectAdapter(
  repoId: string,
  adapterId: string | null
): Promise<void> {
  await api.post(`/repos/${repoId}/lora/select`, { adapter_id: adapterId });
}

export async function startLoRATraining(
  repoId: string
): Promise<{ estimated_minutes: number; num_samples: number }> {
  const { data } = await api.post(`/repos/${repoId}/lora/train`);
  return data;
}

export async function cancelLoRATraining(repoId: string): Promise<void> {
  await api.post(`/repos/${repoId}/lora/cancel`);
}

export async function getLoRAStatus(repoId: string): Promise<LoRAStatus> {
  const { data } = await api.get(`/repos/${repoId}/lora/status`);
  return data;
}
