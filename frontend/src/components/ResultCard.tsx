import type { PredictionResponse } from "../api";
import RiskBadge from "./RiskBadge";

interface Props {
  result: PredictionResponse;
}

export default function ResultCard({ result }: Props) {
  const formatted = new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: result.currency || "INR",
    maximumFractionDigits: 0,
  }).format(result.predicted_cost);

  return (
    <div className="rounded-2xl bg-white p-6 shadow-lg ring-1 ring-slate-200">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-700">
          Estimated insurance cost
        </h2>
        <RiskBadge risk={result.risk_category} />
      </div>
      <p className="mt-2 text-4xl font-bold tracking-tight text-indigo-600">
        {formatted}
      </p>

      {result.top_drivers.length > 0 && (
        <div className="mt-6">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
            Top model drivers
          </h3>
          <ul className="mt-2 space-y-1">
            {result.top_drivers.map((d) => (
              <li
                key={d.feature}
                className="flex justify-between text-sm text-slate-600"
              >
                <span className="font-mono">{d.feature}</span>
                <span className="tabular-nums text-slate-400">
                  {d.importance.toLocaleString()}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <p className="mt-6 text-xs text-slate-400">
        Decision-support estimate only — not a final underwriting decision.
      </p>
    </div>
  );
}
