import { useEffect, useState } from "react";
import {
  getSchema,
  predict,
  type ApplicantPayload,
  type FieldSpec,
  type PredictionResponse,
} from "./api";
import InsuranceForm from "./components/InsuranceForm";
import ResultCard from "./components/ResultCard";

export default function App() {
  const [fields, setFields] = useState<FieldSpec[] | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getSchema()
      .then(setFields)
      .catch(() =>
        setError("Could not load the form schema. Is the API running on :8000?")
      );
  }, []);

  const handleSubmit = async (payload: ApplicantPayload) => {
    setLoading(true);
    setError(null);
    try {
      setResult(await predict(payload));
    } catch (err: unknown) {
      const detail =
        (err as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail ?? "Prediction failed. Check the API and the model artifact.";
      setError(detail);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="bg-gradient-to-r from-indigo-600 to-violet-600 py-10 text-white">
        <div className="mx-auto max-w-5xl px-4">
          <h1 className="text-3xl font-bold tracking-tight">
            Insurance Price Prediction
          </h1>
          <p className="mt-1 text-indigo-100">
            Estimate an applicant&apos;s insurance cost and risk band from health
            &amp; lifestyle inputs.
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-8">
        {error && (
          <div className="mb-6 rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-700 ring-1 ring-rose-200">
            {error}
          </div>
        )}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
          <div className="lg:col-span-3">
            {fields ? (
              <InsuranceForm
                fields={fields}
                loading={loading}
                onSubmit={handleSubmit}
              />
            ) : (
              <div className="rounded-2xl bg-white p-6 text-slate-400 shadow">
                Loading form…
              </div>
            )}
          </div>
          <div className="lg:col-span-2">
            {result ? (
              <ResultCard result={result} />
            ) : (
              <div className="rounded-2xl border-2 border-dashed border-slate-200 p-6 text-center text-slate-400">
                Fill in the form and submit to see the estimated cost.
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
