import { useState } from "react";
import type { ApplicantPayload, FieldSpec } from "../api";

interface Props {
  fields: FieldSpec[];
  loading: boolean;
  onSubmit: (payload: ApplicantPayload) => void;
}

function initialState(fields: FieldSpec[]): ApplicantPayload {
  const state: ApplicantPayload = {};
  for (const f of fields) state[f.name] = f.default;
  return state;
}

export default function InsuranceForm({ fields, loading, onSubmit }: Props) {
  const [values, setValues] = useState<ApplicantPayload>(initialState(fields));

  const update = (name: string, value: string | number | null) =>
    setValues((v) => ({ ...v, [name]: value }));

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(values);
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-2xl bg-white p-6 shadow-lg ring-1 ring-slate-200"
    >
      <h2 className="mb-4 text-lg font-semibold text-slate-700">
        Applicant details
      </h2>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {fields.map((f) => (
          <div key={f.name} className="flex flex-col">
            <label className="mb-1 text-sm font-medium text-slate-600">
              {f.label}
            </label>
            {f.kind === "select" || f.kind === "binary" ? (
              <select
                className="rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                value={values[f.name] ?? ""}
                onChange={(e) =>
                  update(
                    f.name,
                    f.kind === "binary" ? Number(e.target.value) : e.target.value
                  )
                }
              >
                {f.options?.map((opt) => (
                  <option key={String(opt)} value={String(opt)}>
                    {String(opt)}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="number"
                className="rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                value={values[f.name] ?? ""}
                min={f.min}
                max={f.max}
                step={f.step}
                placeholder={f.optional ? "optional" : ""}
                onChange={(e) =>
                  update(
                    f.name,
                    e.target.value === "" ? null : Number(e.target.value)
                  )
                }
              />
            )}
          </div>
        ))}
      </div>
      <button
        type="submit"
        disabled={loading}
        className="mt-6 w-full rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white shadow transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-60"
      >
        {loading ? "Predicting..." : "Predict insurance cost"}
      </button>
    </form>
  );
}
