#!/usr/bin/env node
// =====================================================================
//  🔬 학습 검증 스크립트 — 브라우저 없이 "학습이 실제로 되는지" 확인
//
//  ai-pingpong-rl.html 안의 <script id="rl-core">(물리·규칙·학습 방법)를 그대로 꺼내 실행한다.
//  1) 대조군: 미학습 AI vs 미학습 AI  → 50% 근처여야 측정이 공정
//  2) 선택한 학습 방법으로 학습, 체크포인트마다 '미학습 AI 상대 승률' 측정 → 학습 곡선
//  3) 최종: 학습 AI vs 미학습 AI → 50%에서 통계적으로 유의하게 벗어나면 ✅ (종료 코드 0)
//
//  사용법:  node verify-learning.js [학습량] [최종평가점수=400] [--method q] [--seed 숫자 | --seeds 1,2,3]
//    학습량      방법의 진행 단위 기준(q: 판, evo: 세대). 생략하면 방법마다 기본값
//    --method    학습 방법 id (기본 q). --list 로 목록 보기
//    --seed      시드 하나로 재현 가능한 실행(학습 곡선 전체 출력)
//    --seeds     여러 시드를 차례로 실행해 시드별 최종 결과와 무너진 시드 수를 요약(모든 시드 통과 시 종료 코드 0)
// =====================================================================
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

// --- 인자 ---
const args = process.argv.slice(2);
function takeOpt(name) { const i = args.indexOf(name); if (i < 0) return null; const v = args[i + 1]; args.splice(i, 2); return v; }
function takeFlag(name) { const i = args.indexOf(name); if (i < 0) return false; args.splice(i, 1); return true; }
const methodId = takeOpt('--method') || 'q';
const seedOpt = takeOpt('--seed');
const seedsOpt = takeOpt('--seeds');
const listOnly = takeFlag('--list');

// --- HTML에서 코어 추출 ---
const htmlPath = path.join(__dirname, 'ai-pingpong-rl.html');
const html = fs.readFileSync(htmlPath, 'utf8');
const m = html.match(/<script id="rl-core">([\s\S]*?)<\/script>/);
if (!m) { console.error('❌ ai-pingpong-rl.html에서 <script id="rl-core">를 찾지 못했습니다.'); process.exit(2); }
const ctx = { console };
vm.createContext(ctx);
vm.runInContext(m[1], ctx, { filename: 'rl-core' });
const RL = ctx.PongRL;

if (listOnly) {
  console.log('학습 방법 목록 (--method):');
  for (const d of RL.METHODS) console.log('  %s  %s %s — 기본 학습량 %s', d.id.padEnd(10), d.icon, d.name, d.defaultTrain);
  process.exit(0);
}
const def = RL.getMethod(methodId);
if (!def) { console.error('❌ 알 수 없는 학습 방법: %s (가능: %s)', methodId, RL.METHODS.map(d => d.id).join(', ')); process.exit(2); }

const TRAIN = Number(args[0]) || def.defaultTrain;
const FINAL_POINTS = Number(args[1]) || 400;
const CHECK_POINTS = 100;   // 체크포인트 평가 점수(브라우저 패널과 동일)
const PASS = r => r.z >= 3 && r.winRateA >= 0.7;   // 학습 확인 기준(브라우저 판정과 동일)
const COLLAPSED = r => r.winRateA < 0.6;            // 무너진 시드

function setSeed(seed) {   // 재현 가능한 실행 (mulberry32)
  let s = seed >>> 0;
  RL.setRandom(() => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; });
}

const pct = v => (100 * v).toFixed(1).padStart(5) + '%';
const bar = v => '█'.repeat(Math.round(v * 20)).padEnd(20, '·');

// 한 번 실행: 대조군 → 학습 곡선 → 최종 평가. quiet면 학습 곡선을 찍지 않음
function runOnce(quiet) {
  const log = quiet ? () => {} : console.log;
  const control = RL.evaluate(RL.makeBrain(), RL.makeBrain(), FINAL_POINTS);
  log('\n[대조군] 미학습 vs 미학습 (%d점): 승률 %s · 받기 %s / %s  ← 50%% 근처면 공정',
    control.n, pct(control.winRateA).trim(), pct(control.retA).trim(), pct(control.retB).trim());

  log('\n[학습 곡선] %s 학습하면서 체크포인트마다 미학습 AI와 %d점 대결 (ε=0, 학습 정지 상태로 평가)', def.name, CHECK_POINTS);
  log('    학습량   경험(공)   승률(vs 미학습)                      받기(학습/미학습)   넘기기(학습/미학습)  셀프 랠리');
  const UNTRAINED = RL.makeBrain();
  const L = def.create();
  const marks = def.checkpoints.filter(x => x < TRAIN).concat([TRAIN]);
  for (const mark of marks) {
    L.trainTo(mark);
    const br = L.brain();
    const r = RL.evaluate(br, UNTRAINED, CHECK_POINTS);
    const self = RL.evaluate(br, br, 40);
    log('  %s  %s  %s %s   %s / %s    %s / %s     %s',
      (String(L.progress()) + L.unit).padStart(7), String(L.balls()).padStart(8), pct(r.winRateA), bar(r.winRateA),
      pct(r.retA), pct(r.retB), pct(r.overA), pct(r.overB), self.avgRally.toFixed(1).padStart(5));
  }
  if (def.summary) log('  (%s)', def.summary(L));

  const fin = RL.evaluate(L.brain(), UNTRAINED, FINAL_POINTS);
  log('\n[최종] 학습 로봇(%s · 공 %d개) vs 미학습 로봇 (%d점): 승률 %s · 받기 %s / %s · 넘기기 %s / %s · z=%s',
    L.label(), L.balls(), fin.n, pct(fin.winRateA).trim(), pct(fin.retA).trim(), pct(fin.retB).trim(), pct(fin.overA).trim(), pct(fin.overB).trim(), fin.z.toFixed(1));
  return { control, fin, L };
}

console.log('🤖 로봇팔 탁구 학습 검증  — %s %s (--method %s)%s', def.icon, def.name, def.id, seedOpt != null ? '  seed=' + seedOpt : '');
if (def.id === 'q') {
  console.log('   축별 Q-러닝 TD(0), α=%s, γ=%s, ε %s→%s ×%s/판', RL.CFG.alpha, RL.CFG.gamma, RL.CFG.epsStart, RL.CFG.epsMin, RL.CFG.epsDecay);
}
console.log('   보상: 받으면 %s · 넘기면 %s · 이기면 %s · 지면 %s', RL.CFG.rHit, RL.CFG.rOver, RL.CFG.rWin, RL.CFG.rLose);

if (seedsOpt) {
  // --- 여러 시드 요약 ---
  const seeds = seedsOpt.split(',').map(Number).filter(x => Number.isFinite(x));
  console.log('   학습량 %s · 최종 평가 %d점 · 시드 %s', TRAIN, FINAL_POINTS, seeds.join(', '));
  console.log('\n  시드   경험(공)   최종 승률   받기    넘기기     z    판정');
  const rows = [];
  for (const sd of seeds) {
    setSeed(sd);
    const t0 = Date.now();
    const { fin, L } = runOnce(true);
    rows.push(fin);
    console.log('  %s  %s    %s   %s   %s  %s   %s  (%s초)',
      String(sd).padStart(4), String(L.balls()).padStart(8), pct(fin.winRateA), pct(fin.retA), pct(fin.overA), fin.z.toFixed(1).padStart(5),
      PASS(fin) ? '✅' : COLLAPSED(fin) ? '💥 무너짐' : '🟡', ((Date.now() - t0) / 1000).toFixed(0));
  }
  const mean = k => rows.reduce((a, r) => a + r[k], 0) / Math.max(1, rows.length);
  const passed = rows.filter(PASS).length, collapsed = rows.filter(COLLAPSED).length;
  console.log('\n[요약] 평균 승률 %s · 평균 받기 %s · 통과 %d/%d · 무너진 시드(승률 60%% 미만) %d개',
    pct(mean('winRateA')).trim(), pct(mean('retA')).trim(), passed, rows.length, collapsed);
  process.exit(passed === rows.length ? 0 : 1);
}

if (seedOpt != null) setSeed(Number(seedOpt));
const { fin } = runOnce(false);
const ok = PASS(fin);
console.log(ok
  ? '✅ 학습 확인: 미학습 로봇보다 확실히 강합니다 (50%에서 표준오차 %s배 → 우연으로는 불가능, p<0.001)'
  : '❌ 학습 미확인: 미학습 로봇과 뚜렷한 차이가 없습니다 (z=%s)', fin.z.toFixed(1));
process.exit(ok ? 0 : 1);
