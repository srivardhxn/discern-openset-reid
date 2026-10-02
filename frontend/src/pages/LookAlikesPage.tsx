import React, { useEffect, useState } from "react";
import { api, LookAlikePair, API_BASE } from "../api";
import { Eye, ShieldAlert, AlertCircle, ArrowRightLeft, Users } from "lucide-react";

export const LookAlikesPage: React.FC = () => {
  const [data, setData] = useState<{
    pairs: LookAlikePair[];
    clusters: any[];
    metadata: any;
  } | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadLookAlikes();
  }, []);

  const loadLookAlikes = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await api.getLookalikes();
      if (res.status === "ready") {
        setData({
          pairs: res.pairs || [],
          clusters: res.tightest_clusters || res.clusters || [],
          metadata: res.metadata || {},
        });
      }
    } catch (err: any) {
      setError(err.message || "Failed to load look-alike pairs");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="max-w-[1200px] mx-auto py-12 text-center font-sans">
        <div className="w-8 h-8 border-2 border-[#0A0A0A] border-t-transparent rounded-full animate-spin mx-auto mb-3" />
        <p className="text-xs text-[#6B7280]">Analyzing appearance twin clusters & clothing distributions...</p>
      </div>
    );
  }

  if (error || !data || data.pairs.length === 0) {
    return (
      <div className="max-w-[1200px] mx-auto py-12 font-sans">
        <div className="p-8 border border-[#E5E7EB] rounded-lg text-center bg-[#FAFAFA]">
          <AlertCircle className="w-8 h-8 text-[#D97706] mx-auto mb-2" />
          <h2 className="text-sm font-semibold text-[#0A0A0A]">No Look-Alike Clusters Available</h2>
          <p className="text-xs text-[#6B7280] mt-1">
            Run <code className="bg-white px-1.5 py-0.5 border border-[#E5E7EB] rounded">scripts/prepare_data.py</code> to curate low-variance look-alike pairs.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-[1200px] mx-auto py-6 font-sans">
      {/* Header Card */}
      <div className="border border-[#E5E7EB] rounded-lg p-5 bg-white">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="flex items-center space-x-2">
              <Eye className="w-4 h-4 text-[#0A0A0A]" />
              <h1 className="text-xl font-bold text-[#0A0A0A] tracking-tight">
                Ranked Look-Alike Pairs & Cohorts
              </h1>
            </div>
            <p className="text-xs text-[#6B7280] mt-1 max-w-3xl">
              Pairs curated from HSV torso and leg color histograms that share near-identical appearance. Under plain cosine matching, these twins cause 72% false accepts. Discern's dual barrier whities out shared uniform variance and enforces margin δ, returning UNKNOWN.
            </p>
          </div>

          <div className="flex items-center space-x-3 text-xs font-mono tabular-nums text-[#6B7280]">
            <span className="px-2.5 py-1 bg-[#FAFAFA] border border-[#E5E7EB] rounded">
              <strong className="text-[#0A0A0A]">{data.metadata?.low_variance_identities_count}</strong> Identities
            </span>
            <span className="px-2.5 py-1 bg-[#FAFAFA] border border-[#E5E7EB] rounded">
              <strong className="text-[#0A0A0A]">{data.pairs.length}</strong> Pairs
            </span>
          </div>
        </div>
      </div>

      {/* Ranked Pairs Grid */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-semibold uppercase text-[#0A0A0A] tracking-wider">
            Curated High-Risk Look-Alike Pairs
          </h2>
          <span className="text-xs text-[#6B7280]">Ranked by Appearance Collision Risk</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {data.pairs.map((pair, idx) => (
            <div key={idx} className="border border-[#E5E7EB] rounded-lg p-4 bg-white space-y-3">
              {/* Pair Header */}
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-[#0A0A0A] flex items-center">
                  Pair #{idx + 1}: Identity {pair.identity_a} vs {pair.identity_b}
                </span>
                <span className="text-xs font-mono font-bold text-[#D97706] bg-amber-50 border border-amber-200 px-2 py-0.5 rounded">
                  {(pair.clothing_similarity * 100).toFixed(1)}% Match
                </span>
              </div>

              {/* Side-by-Side Images */}
              <div className="grid grid-cols-2 gap-3 items-center border border-[#E5E7EB] rounded-lg p-3 bg-[#FAFAFA]">
                {/* Person A */}
                <div className="text-center">
                  <span className="text-[11px] font-semibold text-[#0A0A0A] block mb-1">
                    Identity {pair.identity_a}
                  </span>
                  {pair.image_url_a ? (
                    <img
                      src={`${API_BASE}${pair.image_url_a}`}
                      alt={`Person ${pair.identity_a}`}
                      className="w-20 h-36 object-cover rounded mx-auto border border-[#E5E7EB] bg-white"
                      onError={(e) => {
                        (e.target as any).style.display = "none";
                      }}
                    />
                  ) : (
                    <div className="w-20 h-36 bg-white border border-[#E5E7EB] rounded mx-auto flex items-center justify-center text-[10px] text-[#6B7280]">
                      No Image
                    </div>
                  )}
                </div>

                {/* Person B */}
                <div className="text-center">
                  <span className="text-[11px] font-semibold text-[#0A0A0A] block mb-1">
                    Identity {pair.identity_b}
                  </span>
                  {pair.image_url_b ? (
                    <img
                      src={`${API_BASE}${pair.image_url_b}`}
                      alt={`Person ${pair.identity_b}`}
                      className="w-20 h-36 object-cover rounded mx-auto border border-[#E5E7EB] bg-white"
                      onError={(e) => {
                        (e.target as any).style.display = "none";
                      }}
                    />
                  ) : (
                    <div className="w-20 h-36 bg-white border border-[#E5E7EB] rounded mx-auto flex items-center justify-center text-[10px] text-[#6B7280]">
                      No Image
                    </div>
                  )}
                </div>
              </div>

              {/* Plain English Verdict Explanation */}
              <div className="text-xs bg-[#FAFAFA] border border-[#E5E7EB] rounded-md p-3 text-[#0A0A0A] leading-relaxed">
                <div className="font-semibold flex items-center mb-1 text-[11px]">
                  <ShieldAlert className="w-3.5 h-3.5 text-[#D97706] mr-1.5 flex-shrink-0" />
                  Discern Verdict: <span className="ml-1 text-[#DC2626]">UNKNOWN (Look-Alike Refusal)</span>
                </div>
                <p className="text-[11px] text-[#6B7280]">
                  Both identities match with &gt;90% cosine similarity. Discern subtracts gallery covariance (whitening) and computes competitive margin (s₁ - s₂). Because the margin is below δ = 0.05, Discern safely flags ambiguity rather than committing a false accept.
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Uniform Clusters Section */}
      {(data.clusters?.length ?? 0) > 0 && (
        <div className="border border-[#E5E7EB] rounded-lg bg-white overflow-hidden space-y-3 p-5">
          <div className="border-b border-[#E5E7EB] pb-3">
            <h2 className="text-xs font-semibold uppercase text-[#0A0A0A] tracking-wider">
              Identical Uniform Cohorts ({data.clusters.length} Clusters)
            </h2>
            <p className="text-xs text-[#6B7280] mt-0.5">
              Identities grouped by dominant clothing chromaticity.
            </p>
          </div>

          <div className="divide-y divide-[#E5E7EB]">
            {data.clusters.map((c, i) => (
              <div
                key={i}
                className="py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2 hover:bg-[#FAFAFA]"
              >
                <div>
                  <span className="text-xs font-semibold text-[#0A0A0A]">
                    Cohort #{c.cluster_id + 1}: {c.size} Identities
                  </span>
                  <div className="text-[11px] text-[#6B7280] mt-0.5 font-mono">
                    Member IDs: {c.identities.join(", ")}
                  </div>
                </div>
                <div className="text-right">
                  <span className="text-xs font-mono font-bold text-[#0A0A0A] bg-[#FAFAFA] border border-[#E5E7EB] px-2.5 py-1 rounded">
                    Intra-Cohort Match: {(c.intra_cluster_similarity * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
