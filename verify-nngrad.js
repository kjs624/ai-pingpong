#!/usr/bin/env node
// =====================================================================
//  🧮 역전파 검사 — 🎯 정책 경사(Actor-Critic)의 nnGrad()·∇log π를 수치 미분(중앙 차분)과 비교
//
//  ai-pingpong-rl.html 안의 <script id="rl-core">를 그대로 꺼내 실행한다(verify-learning.js와 같은 방식).
//  ① nnGrad: 출력 tanh(Actor)·linear(Critic) 신경망에서 L = Σ dOut·y 의 기울기
//  ② acLogProbGrad: 가우시안 정책 log π(u|s)의 Actor 가중치·log σ 기울기
//  ③ nnStep으로 지도학습(평균제곱오차)이 실제로 손실을 줄이는지(5단계 재사용 예시)
//  각 가중치를 ±h(=1e-5) 움직여 (L(w+h) − L(w−h)) / 2h 와 역전파 값을 비교.
//  상대 오차 = ‖역전파 − 수치‖ / (‖역전파‖ + ‖수치‖) (기울기 벡터 전체). 1e-7 미만이면 통과(종료 코드 0).
//  (가중치 하나씩 나누면 기울기가 거의 0인 칸에서 반올림 오차가 커 보이므로 벡터 노름으로 비교하고, 칸별 최대 절대 오차도 같이 보여 줌)
//
//  사용법:  node verify-nngrad.js [반복 횟수=20]
// =====================================================================
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const html = fs.readFileSync(path.join(__dirname, 'ai-pingpong-rl.html'), 'utf8');
const m = html.match(/<script id="rl-core">([\s\S]*?)<\/script>/);
if (!m) { console.error('❌ <script id="rl-core">를 찾지 못했습니다.'); process.exit(2); }
const ctx = { console };
vm.createContext(ctx);
vm.runInContext(m[1], ctx, { filename: 'rl-core' });
const RL = ctx.PongRL;
const def = RL.getMethod('ac');
if (!def || !def.api) { console.error('❌ 학습 방법 ac(api)가 없습니다.'); process.exit(2); }
const { nnMake, nnForward, nnGrad, nnStep, makeACBrain, acLogProb, acLogProbGrad } = def.api;

let s = 12345;   // 시드 고정
RL.setRandom(() => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; });
const rnd = () => { s = (s * 1103515245 + 12345) >>> 0; return s / 4294967296 * 2 - 1; };   // 입력용 [-1,1)

const REPS = Number(process.argv[2]) || 20, H = 1e-5, TOL = 1e-7;

function check(name, params, f, grad) {
  // params: 숫자 배열(제자리에서 흔듦), f(): 스칼라, grad: 역전파로 구한 같은 길이 기울기
  let dd = 0, aa = 0, nn = 0, worstAbs = 0;
  for (let k = 0; k < params.length; k++) {
    const w0 = params[k];
    params[k] = w0 + H; const fp = f();
    params[k] = w0 - H; const fm = f();
    params[k] = w0;
    const num = (fp - fm) / (2 * H);
    dd += (grad[k] - num) ** 2; aa += grad[k] ** 2; nn += num ** 2;
    worstAbs = Math.max(worstAbs, Math.abs(grad[k] - num));
  }
  return { name, n: params.length, worst: Math.sqrt(dd) / Math.max(1e-12, Math.sqrt(aa) + Math.sqrt(nn)), worstAbs };
}

const results = [];
for (let rep = 0; rep < REPS; rep++) {
  for (const [out, nOut] of [['tanh', 5], ['linear', 1]]) {
    const net = nnMake(16, 12, nOut, out, 0.8, 0.5);
    const x = Array.from({ length: 16 }, rnd), dOut = Array.from({ length: nOut }, rnd);
    const L = () => nnForward(net, x).reduce((a, y, i) => a + dOut[i] * y, 0);
    const g = new Float64Array(net.w.length);
    nnGrad(net, x, dOut, g, 1);
    results.push(check('nnGrad (출력 ' + out + ', 16→12→' + nOut + ')', net.w, L, g));
  }
  // 가우시안 정책: log π(u|s)의 Actor 가중치·log σ 기울기
  const br = makeACBrain();
  for (let k = 0; k < 5; k++) br.logStd[k] = Math.log(0.2 + 0.6 * Math.abs(rnd()));
  for (let k = 0; k < br.actor.w.length; k++) br.actor.w[k] += 0.3 * rnd();   // 출력층이 0 근처만 있지 않게
  const f = Array.from({ length: 16 }, rnd), u = Array.from({ length: 5 }, rnd);
  const ga = new Float64Array(br.actor.w.length), gs = new Float64Array(5);
  acLogProbGrad(br, f, u, ga, gs, 1);
  results.push(check('∇log π — Actor 가중치', br.actor.w, () => acLogProb(br, f, u), ga));
  results.push(check('∇log π — log σ (축 5개)', br.logStd, () => acLogProb(br, f, u), gs));
}

console.log('🧮 역전파 검사 — 역전파 기울기 vs 수치 미분(중앙 차분, h=%s), %d회 반복(무작위 가중치·입력)\n', H, REPS);
console.log('  검사                                  가중치 수   상대 오차(노름)  칸별 최대 절대 오차');
const byName = {};
for (const r of results) { const b = byName[r.name] || (byName[r.name] = { n: r.n, worst: 0, worstAbs: 0, cnt: 0 }); b.worst = Math.max(b.worst, r.worst); b.worstAbs = Math.max(b.worstAbs, r.worstAbs); b.cnt++; }
let ok = true;
for (const [name, b] of Object.entries(byName)) {
  const pass = b.worst < TOL; ok = ok && pass;
  console.log('  %s %s   %s   %s   %s', name.padEnd(34), String(b.n).padStart(6), b.worst.toExponential(2).padStart(12), b.worstAbs.toExponential(2).padStart(12), pass ? '✅' : '❌');
}

// ③ 지도학습 예시(5단계 재사용 방식): 무작위 선생님 신경망의 출력을 학생이 nnGrad + nnStep으로 따라 배움
const teacher = nnMake(16, 12, 5, 'tanh', 0.8, 0.8), student = nnMake(16, 12, 5, 'tanh', 0.6, 0.1);
const data = Array.from({ length: 200 }, () => { const x = Array.from({ length: 16 }, rnd); x[15] = 1; return { x, t: nnForward(teacher, x) }; });
const mse = () => data.reduce((a, { x, t }) => a + nnForward(student, x).reduce((b, y, i) => b + 0.5 * (y - t[i]) ** 2, 0), 0) / data.length;
const before = mse(), g = new Float64Array(student.w.length);
for (let ep = 0; ep < 300; ep++) {
  g.fill(0);
  for (const { x, t } of data) { const y = nnForward(student, x); nnGrad(student, x, y.map((v, i) => v - t[i]), g, 1 / data.length); }
  nnStep(student, g, 0.5);
}
const after = mse(), learned = after < before * 0.2;
ok = ok && learned;
console.log('\n  지도학습(nnGrad + nnStep, 300번): 평균제곱오차 %s → %s  %s', before.toFixed(4), after.toFixed(4), learned ? '✅' : '❌');
console.log(ok ? '\n✅ 역전파가 수치 미분과 일치합니다 (상대 오차 < %s)' : '\n❌ 역전파가 수치 미분과 다릅니다', TOL);
process.exit(ok ? 0 : 1);
