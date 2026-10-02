import React from "react";

interface HorizontalSimilarityBarProps {
  top1Similarity: number;
  top2Similarity: number;
  thresholdTau: number;
  marginDelta: number;
  decision: "ACCEPTED" | "UNKNOWN";
  top1Name?: string;
  top2Name?: string;
}

export const HorizontalSimilarityBar: React.FC<HorizontalSimilarityBarProps> = ({
  top1Similarity,
  top2Similarity,
  thresholdTau,
  marginDelta,
  decision,
  top1Name = "Top-1",
  top2Name = "Top-2",
}) => {
  // Clamp values to [0, 1] for positioning percentage
  const s1Pct = Math.max(0, Math.min(100, top1Similarity * 100));
  const s2Pct = Math.max(0, Math.min(100, top2Similarity * 100));
  const tauPct = Math.max(0, Math.min(100, thresholdTau * 100));
  const marginPct = Math.max(0, s1Pct - s2Pct);
  const marginDeltaPct = marginDelta * 100;

  const isAccepted = decision === "ACCEPTED";
  const s1PassedTau = top1Similarity >= thresholdTau;
  const marginPassedDelta = top1Similarity - top2Similarity >= marginDelta;

  return (
    <div className="space-y-3 font-sans">
      <div className="flex items-center justify-between text-xs text-[#0A0A0A]">
        <span className="font-semibold uppercase tracking-wider text-[11px] text-[#6B7280]">
          Similarity & Decision Boundary
        </span>
        <div className="flex items-center space-x-3 text-[11px] tabular-nums font-mono">
          <span>
            s₁: <strong className="text-[#0A0A0A]">{(top1Similarity * 100).toFixed(1)}%</strong>
          </span>
          {top2Similarity > 0 && (
            <span>
              s₂: <strong className="text-[#6B7280]">{(top2Similarity * 100).toFixed(1)}%</strong>
            </span>
          )}
          <span>
            τ: <strong className="text-[#0A0A0A]">{(thresholdTau * 100).toFixed(1)}%</strong>
          </span>
          <span>
            Δ: <strong className={marginPassedDelta ? "text-[#16A34A]" : "text-[#DC2626]"}>
              {((top1Similarity - top2Similarity) * 100).toFixed(1)}%
            </strong>{" "}
            (req: {(marginDelta * 100).toFixed(1)}%)
          </span>
        </div>
      </div>

      {/* Main Track Container */}
      <div className="relative pt-6 pb-6">
        {/* Scale Axis line */}
        <div className="relative w-full h-3 bg-[#FAFAFA] border border-[#E5E7EB] rounded-full overflow-visible">
          {/* Top-1 Filled Bar */}
          <div
            className={`h-full rounded-full transition-all duration-300 ${
              isAccepted ? "bg-[#16A34A]" : "bg-[#D97706]"
            }`}
            style={{ width: `${s1Pct}%` }}
          />

          {/* Top-2 Marker (Competitor) */}
          {top2Similarity > 0 && (
            <div
              className="absolute top-0 bottom-0 w-1 bg-[#6B7280] rounded z-10"
              style={{ left: `${s2Pct}%` }}
              title={`Top-2 Competitor (${top2Name}): ${(top2Similarity * 100).toFixed(1)}%`}
            >
              {/* Callout Tag Above */}
              <div
                className="absolute -top-6 -translate-x-1/2 bg-[#FAFAFA] border border-[#E5E7EB] px-1.5 py-0.2 text-[9px] font-mono tabular-nums text-[#6B7280] rounded whitespace-nowrap shadow-none"
              >
                s₂ {top2Name}
              </div>
            </div>
          )}

          {/* Vertical Threshold Marker (Tau) */}
          <div
            className="absolute -top-1.5 -bottom-1.5 w-0.5 bg-[#0A0A0A] z-20"
            style={{ left: `${tauPct}%` }}
            title={`Decision Threshold τ = ${(thresholdTau * 100).toFixed(1)}%`}
          >
            {/* Threshold Pin on Top */}
            <div className="absolute -top-5 -translate-x-1/2 bg-[#0A0A0A] text-white px-1.5 py-0.2 text-[9px] font-mono rounded whitespace-nowrap">
              τ {(thresholdTau * 100).toFixed(1)}%
            </div>
          </div>

          {/* Top-1 Marker Pin */}
          <div
            className="absolute -top-1 -bottom-1 w-1 bg-white border-2 border-[#0A0A0A] rounded-full z-15"
            style={{ left: `calc(${s1Pct}% - 2px)` }}
          />
        </div>

        {/* Margin Bracket Below the track */}
        {top2Similarity > 0 && s1Pct > s2Pct && (
          <div
            className="absolute bottom-0 h-3 border-b-2 border-l-2 border-r-2 transition-all duration-200"
            style={{
              left: `${s2Pct}%`,
              width: `${marginPct}%`,
              borderColor: marginPassedDelta ? "#16A34A" : "#DC2626",
            }}
          >
            <div
              className={`absolute top-3 left-1/2 -translate-x-1/2 text-[9px] font-mono tabular-nums font-semibold whitespace-nowrap ${
                marginPassedDelta ? "text-[#16A34A]" : "text-[#DC2626]"
              }`}
            >
              Margin Δ: {((top1Similarity - top2Similarity) * 100).toFixed(1)}%
              {!marginPassedDelta && " (Too close!)"}
            </div>
          </div>
        )}
      </div>

      {/* Axis Scale Markers */}
      <div className="flex justify-between text-[10px] text-[#6B7280] font-mono pt-1">
        <span>0%</span>
        <span>25%</span>
        <span>50%</span>
        <span>75%</span>
        <span>100%</span>
      </div>
    </div>
  );
};
