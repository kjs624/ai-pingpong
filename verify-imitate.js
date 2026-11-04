#!/usr/bin/env node
// =====================================================================
//  🧑‍🏫 모방 학습(5단계) 검사 — ai-pingpong-rl.html의 <script id="rl-core">를 그대로 꺼내 실행
//
//  ① 선생님 혼자의 실력(학생의 상한): 선생님이 팔을 움직이는 로봇 vs 미학습 로봇(절반씩 진영 바꿈)
//  ② 평가에 선생님이 끼지 않는지: 학습한 학생을 EvalJob(evaluate)·학습 정지 Match로 겨루는 동안 선생님 함수 호출 0회
//  ③ 같은 경험(공 수)에서 비교: 🎯 ac(혼자 배우기) · 🧑‍🏫 imitate · 🧑‍🏫 imitate+finetune(모방 → Actor-Critic)
//     공 수 체크포인트마다 미학습 로봇과 평가 → 승률·z, 처음으로 승률 70%·z≥3에 도달한 공 수
//
//  사용법:  node verify-imitate.js [--seeds 1,2,3] [--balls 500,1000,2500,5000] [--points 200] [--methods ac,imitate,imitate+ft]
//  모든 시드에서 imitate가 ac보다 적은 공으로(또는 ac가 못 닿는데) 기준에 닿으면 종료 코드 0
// =====================================================================
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const args = process.argv.slice(2);
function takeOpt(name, def) { const i = args.indexOf(name); if (i < 0) return def; const v = args[i + 1]; args.splice(i, 2); return v; }
const seeds = takeOpt('--seeds', '1,2,3').split(',').map(Number);
const ballMarks = takeOpt('--balls', '250,500,1000,1500,2500,5000').split(',').map(Number);
const POINTS = Number(takeOpt('--points', '200'));
const methods = takeOpt('--methods', 'ac,imitate,imitate+ft').split(',');

const html = fs.readFileSync(path.join(__dirname, 'ai-pingpong-rl.html'), 'utf8');
const m = html.match(/<script id="rl-core">([\s\S]*?)<\/script>/);
if (!m) { console.error('❌ <script id="rl-core">를 찾지 못했습니다.'); process.exit(2); }
const ctx = { console };
vm.createContext(ctx);
vm.runInContext(m[1], ctx, { filename: 'rl-core' });
const RL = ctx.PongRL;
const IMD = RL.getMethod('imitate');
if (!IMD || !IMD.api) { console.error('❌ 학습 방법 imitate(api)가 없습니다.'); process.exit(2); }
const { IM_STATS, makeIMBrain } = IMD.api;

function setSeed(seed) {   // verify-learning.js와 같은 mulberry32
  let s = seed >>> 0;
  RL.setRandom(() => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; });
}
const pct = v => (100 * v).toFixed(1).padStart(5) + '%';
const PASS = r => r.z >= 3 && r.winRateA >= 0.7;

// ① 선생님 혼자: EvalJob은 복사본을 학습 정지(frozen)로 만들어 선생님을 끄므로, 같은 방식(절반씩 진영 교대)을 직접 돌림
function teacherVsUntrained(nPoints) {
  const T = makeIMBrain(); T.betaFix = 1; T.noLearn = true;            // 항상 선생님이 움직임, 학습 없음
  const r = { tw: 0, uw: 0, hits: 0, miss: 0, over: 0, voids: 0, caps: 0 };
  for (let half = 0; half < 2; half++) {
    const U = RL.cloneBrain(RL.makeBrain(), { frozen: true, eps: 0 });
    const mt = half === 0 ? new RL.Match(T, U) : new RL.Match(U, T);
    const pt = half === 0 ? mt.p1 : mt.p2, pu = half === 0 ? mt.p2 : mt.p1;
    let g = 0; while (mt.points - mt.caps - mt.voids < nPoints / 2 && g++ < 5e6) mt.step();
    r.tw += pt.pointsWon; r.uw += pu.pointsWon; r.hits += pt.hits; r.miss += pt.misses; r.over += pt.overs; r.voids += mt.voids; r.caps += mt.caps;
  }
  const n = r.tw + r.uw;
  return { n, win: r.tw / n, ret: r.hits / Math.max(1, r.hits + r.miss), over: r.over / Math.max(1, r.hits), z: (r.tw - n / 2) / Math.sqrt(n / 4), voids: r.voids };
}

function makeLearner(id) {
  if (id === 'imitate+ft') return IMD.create({ finetune: true });
  return RL.getMethod(id).create();
}
function trainToBalls(L, B) { let g = 0; while (L.balls() < B && g++ < 1e6) L.run(300); }

console.log('🧑‍🏫 모방 학습 검사 (rl-core)');
setSeed(1);
const t = teacherVsUntrained(400);
console.log('\n[①] 선생님 혼자 vs 미학습 로봇 (%d점): 승률 %s · 받기 %s · 넘기기 %s · z=%s  ← 학생의 상한',
  t.n, pct(t.win).trim(), pct(t.ret).trim(), pct(t.over).trim(), t.z.toFixed(1));

console.log('\n[③] 같은 경험(공 수)에서 미학습 로봇 상대 %d점 평가 (학습 정지, 평균 행동)', POINTS);
const head = '  방법            시드 ' + ballMarks.map(b => String(b).padStart(13)).join('') + '   기준(70%·z≥3) 첫 도달';
console.log(head);
const first = {};      // first[method][seed] = 공 수 또는 Infinity
let teacherLeak = 0;
for (const id of methods) {
  first[id] = {};
  for (const sd of seeds) {
    setSeed(sd);
    const L = makeLearner(id), U = RL.makeBrain(), cells = [];
    let hit = Infinity;
    for (const B of ballMarks) {
      trainToBalls(L, B);
      const c0 = IM_STATS.teacherCalls;
      const r = RL.evaluate(L.brain(), U, POINTS);                     // ② 평가 중 선생님 호출 수
      teacherLeak += IM_STATS.teacherCalls - c0;
      cells.push(pct(r.winRateA) + ' z' + r.z.toFixed(1).padStart(5));
      if (PASS(r) && hit === Infinity) hit = L.balls();
    }
    first[id][sd] = hit;
    console.log('  %s %s  %s   %s', id.padEnd(14), String(sd).padStart(4), cells.join(''), hit === Infinity ? '못 닿음' : '공 ' + hit + '개');
  }
}

// ② 학습 정지 Match(3D 대결과 같은 구성)에서도 선생님이 안 불리는지
setSeed(7);
const L2 = IMD.create(); trainToBalls(L2, 1000);
const c0 = IM_STATS.teacherCalls;
const duel = new RL.Match(RL.cloneBrain(L2.brain(), { frozen: true, eps: 0 }), RL.cloneBrain(RL.makeBrain(), { frozen: true, eps: 0 }));
for (let i = 0; i < 20000; i++) duel.step();
const duelLeak = IM_STATS.teacherCalls - c0;
console.log('\n[②] 평가(EvalJob) 중 선생님 호출: %d회 · 학습 정지 대결(Match 2만 스텝) 중: %d회 · (학습 중 누적 %d회)', teacherLeak, duelLeak, IM_STATS.teacherCalls);

let ok = teacherLeak === 0 && duelLeak === 0;
if (first.imitate && first.ac) {
  console.log('\n[판정] 같은 공 수에서 imitate가 ac보다 먼저 기준 도달:');
  for (const sd of seeds) {
    const a = first.ac[sd], b = first.imitate[sd], win = b < a;
    if (!win) ok = false;
    console.log('  시드 %d: ac %s · imitate %s %s%s', sd, a === Infinity ? '못 닿음' : a, b === Infinity ? '못 닿음' : b,
      win ? '✅' : '❌', first['imitate+ft'] ? ' · imitate+ft ' + (first['imitate+ft'][sd] === Infinity ? '못 닿음' : first['imitate+ft'][sd]) : '');
  }
}
console.log(ok ? '✅ 통과' : '❌ 미통과');
process.exit(ok ? 0 : 1);
