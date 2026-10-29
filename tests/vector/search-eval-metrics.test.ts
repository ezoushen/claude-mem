import { describe, expect, it } from 'bun:test';
import { qualityBar, scoreRun } from '../../scripts/vector-index/metrics.js';

// Grades: 2 = the note answers the query, 1 = related, 0 or absent = not relevant.
const qrels = {
  q1: { 10: 2, 11: 1 },
  q2: { 20: 2 },
  q3: { 30: 1 }, // no grade-2 note: left out of every score
};
const kinds = { q1: 'natural', q2: 'agent', q3: 'agent' };

describe('scoreRun', () => {
  it('computes nDCG@10, S@5 and MRR per query group the way the published eval did', () => {
    // q1 ranks the related note first: DCG = 1/log2(2) + 3/log2(3), ideal = 3/log2(2) + 1/log2(3).
    // q2 finds its note at rank 6: outside the top 5, reciprocal rank 1/6.
    const run = { q1: [11, 10], q2: [1, 2, 3, 4, 5, 20], q3: [30] };
    const q1 = (1 + 3 / Math.log2(3)) / (3 + 1 / Math.log2(3));
    const q2 = 3 / Math.log2(7) / 3;
    const s = scoreRun(run, qrels, kinds);
    expect(s.all.n).toBe(2);
    expect(s.all.ndcg).toBeCloseTo((q1 + q2) / 2, 10);
    expect(s.all.s5).toBeCloseTo(0.5, 10);
    expect(s.all.mrr).toBeCloseTo((1 / 2 + 1 / 6) / 2, 10);
    expect(s.natural).toEqual({ ndcg: expect.closeTo(q1, 10), s5: 1, mrr: 0.5, n: 1 });
    expect(s.agent.n).toBe(1);
  });
});

describe('qualityBar', () => {
  it('fails a variant that loses more than 0.02 overall or 0.05 in any group', () => {
    const base = { all: { ndcg: 0.70, s5: 0, mrr: 0, n: 1 }, agent: { ndcg: 0.70, s5: 0, mrr: 0, n: 1 } };
    expect(qualityBar(base, { all: { ...base.all, ndcg: 0.69 }, agent: { ...base.agent, ndcg: 0.66 } })).toEqual({ pass: true, reasons: [] });
    const worse = qualityBar(base, { all: { ...base.all, ndcg: 0.67 }, agent: { ...base.agent, ndcg: 0.64 } });
    expect(worse.pass).toBe(false);
    expect(worse.reasons).toHaveLength(2);
  });
});
