/**
 * Reciprocal-rank fusion: each list adds 1 / (k + rank) to the rows it holds,
 * rank counted from 0, and rows are ordered by the sum. k = 60 is the value of
 * the original paper and of the evaluation behind this module (#3610), where
 * fusing keyword search with bge-m3 ranked best (nDCG@10 0.71, against 0.40
 * for keyword search and 0.66 for bge-m3 alone). A row in both lists keeps
 * the object from the first list it appears in; ties keep first-seen order.
 */
export const RRF_K = 60;

export function reciprocalRankFusion<T extends { id: number }>(lists: T[][], k: number = RRF_K): T[] {
  const fused = new Map<number, { row: T; score: number; order: number }>();
  for (const list of lists) {
    list.forEach((row, rank) => {
      const entry = fused.get(row.id);
      if (entry) entry.score += 1 / (k + rank);
      else fused.set(row.id, { row, score: 1 / (k + rank), order: fused.size });
    });
  }
  return [...fused.values()]
    .sort((a, b) => b.score - a.score || a.order - b.order)
    .map(entry => entry.row);
}
