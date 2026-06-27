import axios from "axios";

const BASE = import.meta.env.VITE_API_URL ?? "/api";

export interface FieldSpec {
  name: string;
  label: string;
  kind: "number" | "select" | "binary";
  options?: (string | number)[];
  min?: number;
  max?: number;
  step?: number;
  default: string | number | null;
  optional?: boolean;
}

export interface Driver {
  feature: string;
  importance: number;
}

export interface PredictionResponse {
  predicted_cost: number;
  risk_category: "Low" | "Medium" | "High";
  currency: string;
  top_drivers: Driver[];
}

export type ApplicantPayload = Record<string, string | number | null>;

export async function getSchema(): Promise<FieldSpec[]> {
  const { data } = await axios.get<{ fields: FieldSpec[] }>(`${BASE}/schema`);
  return data.fields;
}

export async function predict(
  payload: ApplicantPayload
): Promise<PredictionResponse> {
  const { data } = await axios.post<PredictionResponse>(
    `${BASE}/predict`,
    payload
  );
  return data;
}

export async function getHealth(): Promise<{
  status: string;
  model_loaded: boolean;
  model_name?: string;
}> {
  const { data } = await axios.get(`${BASE}/health`);
  return data;
}
