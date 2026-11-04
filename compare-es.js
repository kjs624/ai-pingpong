#!/usr/bin/env node
// =====================================================================
//  🧭 3단계 검증: 같은 경험(공 수)에서 진화 전략(es)과 유전 알고리즘(evo)의 볼 머신 점수 비교
//
//  공 수 체크포인트마다 각 방법의 bestBrain()을 따로 떼어
//    · 볼 머신 테스트 K번(최대 40공)의 평균 점수(선택용 가산점 없이 화면 점수 그대로)
//  를 재고, 마지막 체크포인트에서는 미학습 로봇과 P점 대결 승률도 잰다.
//  평가는 학습과 다른 난수열(시드·체크포인트마다 고정, 두 방법이 같은 공을 받음)을 써서 학습 경로에 영향을 주지 않는다.
//
//  사용법: node compare-es.js [--seeds 1,2,3] [--balls 2000,5000,10000,20000,30000] [--k 60] [--points 200]
//                             [--methods evo,es] [--sigma 0.1] [--alpha 0.05] [--trials 12] [--init best|rand] [--pick brain|top]
// =====================================================================
'use strict';
const fs = require('fs'), path = require('path'), vm = require('vm');
const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf(n); return i < 0 ? d : args[i + 1]; };
const seeds = opt('--seeds', '1,2,3').split(',').map(Number);
const marks = opt('--balls', '2000,5000,10000,20000,30000').split(',').map(Number);
const K = Number(opt('--k', 60)), PTS = Number(opt('--points', 200));
const methods = opt('--methods', 'evo,es').split(',');
const esOpt = {}; if (opt('--sigma')) esOpt.sigma = Number(opt('--sigma')); if (opt('--alpha')) esOpt.alpha = Number(opt('--alpha')); if (opt('--trials')) esOpt.trials = Number(opt('--trials')); if (opt('--init')) esOpt.init = opt('--init');
const pick = opt('--pick', 'brain');   // es만: brain = bestBrain()(중심), top = 그 세대 최고 개체

const html = fs.readFileSync(path.join(__dirname, 'ai-pingpong-rl.html'), 'utf8');
const ctx = { console }; vm.createContext(ctx);
vm.runInContext(html.match(/<script id="rl-core">([\s\S]*?)<\/script>/)[1], ctx);
const RL = ctx.PongRL;

function mulberry(seed) { let s = seed >>> 0; return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function drillScore(brain, k) {   // 볼 머신 테스트 k번 평균 점수 · 평균 연속
  let sum = 0, streak = 0;
  for (let t = 0; t < k; t++) {
    const m = new RL.Match(RL.makeNNBrain(brain.w.slice()), RL.MACHINE, null, { drill: { maxBalls: 40 } });
    let guard = 0; while (!m.drill.done && guard++ < 2e6) m.step();
    sum += m.drill.lastScore; streak += m.drill.lastStreak;
  }
  return { score: sum / k, streak: streak / k };
}
// 학습 난수(train)와 평가 난수(eval)를 갈아 끼움
function withRand(r, f) { RL.setRandom(r); return f(); }

const res = {};   // res[method][seed] = [{balls, gen, score, streak, win?}]
for (const id of methods) {
  const def = RL.getMethod(id); res[id] = {};
  for (const sd of seeds) {
    const train = mulberry(sd); RL.setRandom(train);
    const L = def.create(id === 'es' ? esOpt : undefined), rows = [];
    const t0 = Date.now();
    for (const b of marks) {
      RL.setRandom(train); while (L.balls() < b) L.run(300);
      const br = pick === 'top' && L.evo.best ? RL.makeNNBrain(L.evo.best.slice()) : L.brain();
      const ev = withRand(mulberry(1000 * sd + b), () => drillScore(br, K));
      const row = { balls: L.balls(), gen: L.progress(), score: ev.score, streak: ev.streak };
      if (b === marks[marks.length - 1]) row.win = withRand(mulberry(7 * sd + 3), () => RL.evaluate(br, RL.makeBrain(), PTS));
      rows.push(row);
    }
    res[id][sd] = rows;
    const f = rows.at(-1);
    console.log('%s seed %d: %s  · 최종 vs 미학습 승률 %s%% 받기 %s%% z=%s (%ss)', id.padEnd(3), sd,
      rows.map(r => r.balls + '공/' + r.gen + '세대 ' + r.score.toFixed(2)).join(' | '),
      (100 * f.win.winRateA).toFixed(1), (100 * f.win.retA).toFixed(1), f.win.z.toFixed(1), ((Date.now() - t0) / 1000).toFixed(0));
  }
}
// 표: 체크포인트별 시드 평균 볼 머신 점수
console.log('\n| 공 수 | ' + methods.map(id => id + ' 점수(시드 ' + seeds.join('·') + ')').join(' | ') + ' | ' + methods.map(id => id + ' 평균').join(' | ') + ' |');
console.log('|' + ' --- |'.repeat(1 + 2 * methods.length));
marks.forEach((b, j) => {
  const per = methods.map(id => seeds.map(sd => res[id][sd][j].score.toFixed(2)).join(' / '));
  const mean = methods.map(id => (seeds.reduce((a, sd) => a + res[id][sd][j].score, 0) / seeds.length).toFixed(2));
  console.log('| ' + b + ' | ' + per.join(' | ') + ' | ' + mean.join(' | ') + ' |');
});
const winMean = id => seeds.reduce((a, sd) => a + res[id][sd].at(-1).win.winRateA, 0) / seeds.length;
console.log('| 최종 vs 미학습 승률 (' + PTS + '점) | ' + methods.map(id => seeds.map(sd => (100 * res[id][sd].at(-1).win.winRateA).toFixed(1) + '%').join(' / ')).join(' | ') +
  ' | ' + methods.map(id => (100 * winMean(id)).toFixed(1) + '%').join(' | ') + ' |');
const auc = id => seeds.reduce((a, sd) => a + res[id][sd].reduce((s, r) => s + r.score, 0) / marks.length, 0) / seeds.length;
console.log('\n체크포인트 평균(곡선 아래 넓이/개수): ' + methods.map(id => id + ' ' + auc(id).toFixed(2)).join(' · '));
