#!/usr/bin/env node
// =====================================================================
//  🔬 학습 검증 스크립트 — 브라우저 없이 "강화학습이 실제로 되는지" 확인
//
//  ai-pingpong-rl.html 안의 <script id="rl-core">(물리·규칙·Q-러닝)를 그대로 꺼내 실행한다.
//  1) 대조군: 미학습 AI vs 미학습 AI  → 50% 근처여야 측정이 공정
//  2) 빈 뇌로 셀프플레이 학습, 체크포인트마다 '미학습 AI 상대 승률' 측정 → 학습 곡선
//  3) 최종: 학습 AI vs 미학습 AI → 50%에서 통계적으로 유의하게 벗어나면 ✅ (종료 코드 0)
//
//  사용법:  node verify-learning.js [학습판수=1000] [최종평가점수=400] [--seed 숫자]
// =====================================================================
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const args = process.argv.slice(2);
const seedIdx = args.indexOf('--seed');
const seed = seedIdx >= 0 ? Number(args.splice(seedIdx, 2)[1]) : null;
const TRAIN = Number(args[0]) || 1000;
const FINAL_POINTS = Number(args[1]) || 400;
const CHECK_POINTS = 100;   // 체크포인트 평가 점수(브라우저 패널과 동일)

// --- HTML에서 강화학습 코어 추출 ---
const htmlPath = path.join(__dirname, 'ai-pingpong-rl.html');
const html = fs.readFileSync(htmlPath, 'utf8');
const m = html.match(/<script id="rl-core">([\s\S]*?)<\/script>/);
if (!m) { console.error('❌ ai-pingpong-rl.html에서 <script id="rl-core">를 찾지 못했습니다.'); process.exit(2); }
const ctx = { console };
vm.createContext(ctx);
vm.runInContext(m[1], ctx, { filename: 'rl-core' });
const RL = ctx.PongRL;

if (seed != null) {   // 재현 가능한 실행 (mulberry32)
  let s = seed >>> 0;
  RL.setRandom(() => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; });
}

const pct = v => (100 * v).toFixed(1).padStart(5) + '%';
const bar = v => '█'.repeat(Math.round(v * 20)).padEnd(20, '·');

console.log('🏓 AI 탁구 강화학습 검증  (Q-러닝 TD(0), α=%s, γ=%s/%s, ε %s→%s ×%s/판)%s',
  RL.CFG.alpha, RL.CFG.gammaMove, RL.CFG.gammaShot, RL.CFG.epsStart, RL.CFG.epsMin, RL.CFG.epsDecay,
  seed != null ? '  seed=' + seed : '');

// 1) 대조군
const control = RL.evaluate(RL.makeBrain(), RL.makeBrain(), FINAL_POINTS);
console.log('\n[대조군] 미학습 vs 미학습 (%d점): 승률 %s · 리턴 %s / %s  ← 50%% 근처면 공정',
  control.n, pct(control.winRateA).trim(), pct(control.retA).trim(), pct(control.retB).trim());

// 2) 학습 곡선
console.log('\n[학습 곡선] 셀프플레이로 학습하면서 체크포인트마다 미학습 AI와 %d점 대결 (ε=0, 학습 정지 상태로 평가)', CHECK_POINTS);
console.log('  학습판수   승률(vs 미학습)                      리턴(학습/미학습)  셀프 랠리');
const UNTRAINED = RL.makeBrain();
const brain = RL.makeBrain();
const marks = [0, 25, 50, 100, 200, 400, 700, 1000, 1500, 2000, 3000, 5000, 10000].filter(x => x < TRAIN).concat([TRAIN]);
let match = null;
for (const mark of marks) {
  match = RL.trainSelfPlay(brain, mark - brain.episodes, match);
  const r = RL.evaluate(brain, UNTRAINED, CHECK_POINTS);
  const self = RL.evaluate(brain, brain, 40);
  console.log('  %s판  %s %s   %s / %s     %s',
    String(brain.episodes).padStart(6), pct(r.winRateA), bar(r.winRateA),
    pct(r.retA), pct(r.retB), self.avgRally.toFixed(1).padStart(5));
}
console.log('  (Q-테이블 상태 수: 이동 %d · 샷 %d, TD 업데이트 %d회)',
  Object.keys(brain.move).length, Object.keys(brain.shot).length, brain.updates);

// 3) 최종 판정
const fin = RL.evaluate(brain, UNTRAINED, FINAL_POINTS);
console.log('\n[최종] 학습 AI(%d판) vs 미학습 AI (%d점): 승률 %s · 리턴 %s / %s · z=%s',
  brain.episodes, fin.n, pct(fin.winRateA).trim(), pct(fin.retA).trim(), pct(fin.retB).trim(), fin.z.toFixed(1));
const ok = fin.z >= 3 && fin.winRateA >= 0.7;
console.log(ok
  ? '✅ 학습 확인: 미학습 AI보다 확실히 강합니다 (50%에서 표준오차 %s배 → 우연으로는 불가능, p<0.001)'
  : '❌ 학습 미확인: 미학습 AI와 뚜렷한 차이가 없습니다 (z=%s)', fin.z.toFixed(1));
process.exit(ok ? 0 : 1);
