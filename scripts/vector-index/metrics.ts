/**
 * Scoring for search evaluations, as the 2026-10-28 evaluation behind the
 * local vector index did it (thedotmack/claude-mem#3610): graded relevance
 * 0 / 1 / 2 per (query, observation), nDCG@10 with gain 2^grade - 1, S@5 (a
 * grade-2 note in the top five), MRR of the first grade-2 note, per query group.
 * Queries with no grade-2 note are left out, as there is nothing to find.
 * (Fusion itself is src/services/vector/rrf.ts.)
 */

export interface Score { ndcg: number; s5: number; mrr: number; n: number }
/** Grades per query id, per observation id. */
export type Qrels = Record<string, Record<string, number>>;

const gain = (grade: number) => 2 ** grade - 1;
const gradeOf = (grades: Record<string, number>, id: number) => grades[String(id)] ?? 0;

function ndcg(ids: number[], grades: Record<string, number>): number {
  const dcg = ids.slice(0, 10).reduce((sum, id, rank) => sum + gain(gradeOf(grades, id)) / Math.log2(rank + 2), 0);
  const ideal = Object.values(grades).sort((a, b) => b - a).slice(0, 10)
    .reduce((sum, grade, rank) => sum + gain(grade) / Math.log2(rank + 2), 0);
  return dcg / ideal;
}

function firstAnswer(ids: number[], grades: Record<string, number>): number {
  return ids.slice(0, 10).findIndex(id => gradeOf(grades, id) === 2);
}

/** Mean scores over all queries (`all`) and per group named in `kinds`. */
export function scoreRun(run: Record<string, number[]>, qrels: Qrels, kinds: Record<string, string>): Record<string, Score> {
  const answerable = Object.keys(run).filter(q => qrels[q] && Object.values(qrels[q]).some(grade => grade === 2));
  const groups = ['all', ...new Set(answerable.map(q => kinds[q]).filter(Boolean))];
  const out: Record<string, Score> = {};
  for (const group of groups) {
    const qs = answerable.filter(q => group === 'all' || kinds[q] === group);
    const mean = (f: (q: string) => number) => qs.reduce((sum, q) => sum + f(q), 0) / qs.length;
    out[group] = {
      ndcg: mean(q => ndcg(run[q], qrels[q])),
      s5: mean(q => { const r = firstAnswer(run[q], qrels[q]); return r >= 0 && r < 5 ? 1 : 0; }),
      mrr: mean(q => { const r = firstAnswer(run[q], qrels[q]); return r < 0 ? 0 : 1 / (r + 1); }),
      n: qs.length,
    };
  }
  return out;
}

/**
 * The bar every cheaper variant had to clear against full precision: fused
 * nDCG@10 no more than 0.02 lower overall, and no group more than 0.05 lower.
 */
export function qualityBar(base: Record<string, Score>, variant: Record<string, Score>): { pass: boolean; reasons: string[] } {
  const reasons: string[] = [];
  for (const group of Object.keys(base)) {
    const limit = group === 'all' ? 0.02 : 0.05;
    const drop = base[group].ndcg - (variant[group]?.ndcg ?? 0);
    if (drop > limit + 1e-9) {
      reasons.push(`${group}: nDCG@10 ${(variant[group]?.ndcg ?? 0).toFixed(3)} vs ${base[group].ndcg.toFixed(3)} (drop ${drop.toFixed(3)} > ${limit})`);
    }
  }
  return { pass: reasons.length === 0, reasons };
}
