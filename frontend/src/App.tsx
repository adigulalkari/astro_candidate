import React, { useState } from 'react';
import { Search, Loader2, CheckCircle2, AlertTriangle, BookOpen, Terminal } from 'lucide-react';

// --- Types based on Backend Specification ---
interface Candidate {
  planet: string;
  score: number;
  reasons: string[];
}

interface Evidence {
  title: string;
  year: number;
  url: string;
  claim_supported: string;
}

interface ChatResponse {
  answer: string;
  steps_log: string[];
  candidates: Candidate[];
  evidence: Evidence[];
  uncertainties: string[];
}

export default function App() {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [data, setData] = useState<ChatResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      // Assumes your FastAPI backend is running on port 8000
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) throw new Error('Failed to fetch data from agent');
      
      const result: ChatResponse = await response.json();
      setData(result);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-300 font-sans p-6 selection:bg-indigo-500/30">
      <header className="mb-8 border-b border-gray-800 pb-4">
        <h1 className="text-2xl font-light text-gray-100 tracking-tight">Exoplanet Candidate Analyst</h1>
        <p className="text-sm text-gray-500 mt-1">Agentic ML & RAG System Interface</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* LEFT COLUMN: Agent Interaction */}
        <div className="lg:col-span-4 space-y-6">
          <form onSubmit={handleSearch} className="space-y-4">
            <div className="relative">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., Find rocky exoplanets within 50 light years and show supporting literature."
                className="w-full h-32 bg-gray-900 border border-gray-800 rounded-none p-4 text-gray-200 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-colors resize-none placeholder:text-gray-600"
              />
            </div>
            <button
              type="submit"
              disabled={isLoading}
              className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-3 text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
              {isLoading ? 'Analyzing...' : 'Execute Agent Query'}
            </button>
          </form>

          {error && (
            <div className="p-4 bg-red-950/30 border border-red-900/50 text-red-400 text-sm">
              {error}
            </div>
          )}

          {/* Agent Steps Log */}
          {data?.steps_log && (
            <div className="border border-gray-800 bg-gray-900/50 p-4">
              <div className="flex items-center gap-2 mb-4 text-gray-400 border-b border-gray-800 pb-2">
                <Terminal className="w-4 h-4" />
                <h3 className="text-xs font-semibold uppercase tracking-wider">Execution Log</h3>
              </div>
              <ul className="space-y-2 text-xs font-mono text-gray-500">
                {data.steps_log.map((step, idx) => (
                  <li key={idx}>&gt; {step}</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* RIGHT COLUMN: Results Dashboard */}
        <div className="lg:col-span-8 space-y-6">
          {data ? (
            <>
              {/* Synthesized Answer */}
              <div className="bg-gray-900 p-6 border border-gray-800">
                <h2 className="text-lg font-medium text-gray-100 mb-4">Synthesis</h2>
                <p className="text-gray-300 leading-relaxed text-sm">{data.answer}</p>
              </div>

              {/* Data Grid: Candidates & Details */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                {/* Candidates List */}
                <div className="space-y-4">
                  <div className="flex items-center gap-2 text-indigo-400">
                    <CheckCircle2 className="w-5 h-5" />
                    <h3 className="text-sm font-semibold uppercase tracking-wider">Prioritized Candidates</h3>
                  </div>
                  {data.candidates.map((candidate, idx) => (
                    <div key={idx} className="border border-gray-800 p-4 bg-gray-900/30 hover:border-gray-700 transition-colors">
                      <div className="flex justify-between items-start mb-3">
                        <span className="text-lg font-medium text-gray-100">{candidate.planet}</span>
                        <span className="text-xs font-mono bg-indigo-900/50 text-indigo-300 px-2 py-1 border border-indigo-800">
                          Score: {candidate.score.toFixed(4)}
                        </span>
                      </div>
                      <ul className="space-y-1">
                        {candidate.reasons.map((reason, rIdx) => (
                          <li key={rIdx} className="text-xs text-gray-400 flex items-start gap-2">
                            <span className="text-gray-600 mt-0.5">•</span> {reason}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>

                {/* Evidence & Uncertainties */}
                <div className="space-y-6">
                  {/* Literature Evidence */}
                  <div className="space-y-4">
                    <div className="flex items-center gap-2 text-emerald-400">
                      <BookOpen className="w-5 h-5" />
                      <h3 className="text-sm font-semibold uppercase tracking-wider">Literature Evidence</h3>
                    </div>
                    {data.evidence.map((ev, idx) => (
                      <div key={idx} className="border-l-2 border-emerald-800 pl-4 py-1">
                        <a href={ev.url} target="_blank" rel="noreferrer" className="text-sm text-gray-200 hover:text-emerald-400 transition-colors">
                          {ev.title} ({ev.year})
                        </a>
                        <p className="text-xs text-gray-500 mt-1 italic">"{ev.claim_supported}"</p>
                      </div>
                    ))}
                  </div>

                  {/* Uncertainties */}
                  {data.uncertainties && data.uncertainties.length > 0 && (
                    <div className="space-y-4 pt-4 border-t border-gray-800">
                      <div className="flex items-center gap-2 text-amber-500">
                        <AlertTriangle className="w-5 h-5" />
                        <h3 className="text-sm font-semibold uppercase tracking-wider">Uncertainties</h3>
                      </div>
                      <ul className="space-y-2">
                        {data.uncertainties.map((unc, idx) => (
                          <li key={idx} className="text-xs text-gray-400 bg-amber-950/20 p-2 border border-amber-900/30">
                            {unc}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

              </div>
            </>
          ) : (
            /* Empty State */
            <div className="h-full min-h-[400px] flex items-center justify-center border border-gray-800 border-dashed">
              <p className="text-gray-600 text-sm">Awaiting agent execution...</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}