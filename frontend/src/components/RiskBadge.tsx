interface Props {
  risk: "Low" | "Medium" | "High";
}

const STYLES: Record<Props["risk"], string> = {
  Low: "bg-emerald-100 text-emerald-700 ring-emerald-600/20",
  Medium: "bg-amber-100 text-amber-700 ring-amber-600/20",
  High: "bg-rose-100 text-rose-700 ring-rose-600/20",
};

export default function RiskBadge({ risk }: Props) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-semibold ring-1 ring-inset ${STYLES[risk]}`}
    >
      {risk} risk
    </span>
  );
}
