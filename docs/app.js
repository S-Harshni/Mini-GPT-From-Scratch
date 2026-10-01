'use strict';
const NS = 'http://www.w3.org/2000/svg';
function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === '') continue;
    if (k === 'text') e.textContent = v; else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v);
  }
  e.append(...kids.flat().filter(k => k != null && k !== ''));
  return e;
}
function s(tag, attrs, parent, text) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  parent.append(e);
  return e;
}
const nf = new Intl.NumberFormat('en-GB');

// Byte-pair encoding in the browser, replaying the merges the Python tokenizer learned.
function makeTokenizer(merges) {
  const rank = new Map(merges.map(([a, b], i) => [a + ',' + b, i]));
  const vocab = [...Array(256)].map((_, i) => [i]);
  merges.forEach(([a, b]) => vocab.push(vocab[a].concat(vocab[b])));
  const bytes = new TextEncoder(), text = new TextDecoder();
  function word(w) {
    let ids = [...bytes.encode(w)];
    while (ids.length > 1) {
      let best = Infinity, at = -1;
      for (let i = 0; i < ids.length - 1; i++) { const r = rank.get(ids[i] + ',' + ids[i + 1]); if (r !== undefined && r < best) { best = r; at = i; } }
      if (at < 0) break;
      const a = ids[at], b = ids[at + 1], out = [];
      for (let i = 0; i < ids.length; i++) { if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) { out.push(256 + best); i++; } else out.push(ids[i]); }
      ids = out;
    }
    return ids;
  }
  return { encode: t => (t.match(/\s*\S+|\s+$/g) || []).flatMap(word), show: id => text.decode(new Uint8Array(vocab[id])) };
}

function lossChart(box, curve) {
  const W = box.clientWidth, H = 260, m = { l: 44, r: 14, t: 12, b: 28 };
  box.replaceChildren();
  const root = s('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img', 'aria-label': 'Training and validation loss' }, box);
  const top = Math.ceil(Math.max(...curve.map(c => c.val))), last = curve.at(-1).step;
  const x = v => m.l + (W - m.l - m.r) * v / last, y = v => m.t + (H - m.t - m.b) * (1 - v / top);
  for (let t = 0; t <= top; t += top > 4 ? 2 : 1) {
    s('line', { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: t ? 'grid-line' : 'axis-line' }, root);
    s('text', { x: m.l - 8, y: y(t) + 4, class: 'tick', 'text-anchor': 'end' }, root, t);
  }
  for (let t = 0; t <= last; t += 1000) s('text', { x: x(t), y: H - 8, class: 'tick', 'text-anchor': 'middle' }, root, nf.format(t));
  for (const [key, color] of [['train', 'var(--series-1)'], ['val', 'var(--series-2)']]) {
    s('path', { d: curve.map((c, i) => `${i ? 'L' : 'M'}${x(c.step).toFixed(1)},${y(c[key]).toFixed(1)}`).join(''), class: 'line', style: `stroke:${color}` }, root);
  }
}

(async function main() {
  const D = await (await fetch('data.json')).json();
  const tok = makeTokenizer(D.merges), c = D.config;
  document.getElementById('lede').textContent =
    `A ${nf.format(D.parameters)}-parameter transformer language model with its own byte-pair tokenizer, written and trained from scratch in PyTorch on a laptop CPU.`;
  document.getElementById('foot').textContent = `Training text: ${D.corpus.name} (public domain), ${nf.format(D.corpus.characters)} characters. Built by S Harshni.`;

  const input = h('textarea', { 'aria-label': 'Text to tokenize' });
  input.value = 'To be, or not to be: that is the question.';
  const out = h('div', { class: 'tokens' }), count = h('p', { class: 'sub', style: 'margin-top:8px' });
  const retokenize = () => {
    const ids = tok.encode(input.value);
    out.replaceChildren(...ids.map(i => h('span', { title: 'token ' + i, text: tok.show(i) })));
    count.textContent = `${ids.length} tokens for ${input.value.length} characters (${(input.value.length / Math.max(ids.length, 1)).toFixed(2)} characters per token).`;
  };
  input.addEventListener('input', retokenize);

  const chart = h('div', { class: 'plot' });
  const rows = [['Unigram counts', D.baselines.unigram], ['Bigram counts', D.baselines.bigram], ['This transformer', { loss: D.best.val_loss, perplexity: D.best.perplexity }]];
  const table = h('table', {}, h('thead', {}, h('tr', {}, ['Model', 'Validation loss (nats per token)', 'Perplexity'].map((t, i) => h('th', { class: i ? 'num' : '', text: t })))),
    h('tbody', {}, rows.map(([n, v], i) => h('tr', { class: i === 2 ? 'best' : '' }, h('td', { text: n }), h('td', { class: 'num', text: v.loss.toFixed(3) }), h('td', { class: 'num', text: v.perplexity.toFixed(1) })))));

  const A = D.attention, heat = h('div', { class: 'table-box' });
  const pick = h('select', { 'aria-label': 'Layer and head' }, A.weights.flatMap((layer, l) => layer.map((_, k) => h('option', { value: l + ',' + k, text: `Layer ${l + 1}, head ${k + 1}` }))));
  const drawHeat = () => {
    const [l, k] = pick.value.split(',').map(Number), w = A.weights[l][k];
    heat.replaceChildren(h('table', { class: 'att' },
      h('thead', {}, h('tr', {}, h('th', {}), A.tokens.map(t => h('th', { text: t.replace(/\n/g, '\\n') })))),
      h('tbody', {}, w.map((row, i) => h('tr', {}, h('th', { text: A.tokens[i].replace(/\n/g, '\\n') }),
        row.map((v, j) => h('td', { title: j <= i ? `${v.toFixed(2)}` : 'masked', style: j <= i ? `background:color-mix(in srgb, var(--series-1) ${Math.round(v * 100)}%, var(--surface))` : '' })))))));
  };
  pick.addEventListener('change', drawHeat);

  document.getElementById('main').append(
    h('div', { class: 'tiles' }, [
      ['Parameters', nf.format(D.parameters), `${c.layers} layers, ${c.heads} heads, width ${c.width}, context ${c.context}`],
      ['Validation perplexity', D.best.perplexity.toFixed(1), `${D.baselines.bigram.perplexity.toFixed(1)} for a bigram model`],
      ['Vocabulary', nf.format(c.vocab_size), `byte-pair tokens, ${D.corpus.characters_per_token} characters each on average`],
      ['Training', `${Math.round(D.training.seconds / 60)} min`, `${nf.format(D.training.steps)} steps on CPU, best at step ${nf.format(D.best.step)}`],
    ].map(([l, v, n]) => h('div', { class: 'tile' }, h('div', { class: 'label', text: l }), h('div', { class: 'value', text: v }), h('div', { class: 'note', text: n })))),
    h('div', { class: 'grid' },
      h('figure', { class: 'card wide' }, h('div', { class: 'card-head' }, h('div', {}, h('h3', { text: 'Tokenizer playground' }), h('p', { class: 'sub', text: 'Type anything. The text is split by the byte-pair merges learned from the training text; hover a token for its id.' }))), input, out, count),
      h('figure', { class: 'card' }, h('div', { class: 'card-head' }, h('div', {}, h('h3', { text: 'Loss during training' }), h('p', { class: 'sub', text: 'Cross-entropy by step. The gap that opens between the lines is the model starting to memorise a small corpus.' }))),
        h('div', { class: 'legend' }, h('span', {}, h('i', { style: 'background:var(--series-1)' }), 'Training'), h('span', {}, h('i', { style: 'background:var(--series-2)' }), 'Validation')), chart),
      h('figure', { class: 'card' }, h('div', { class: 'card-head' }, h('div', {}, h('h3', { text: 'Against counting baselines' }), h('p', { class: 'sub', text: 'On held-out text. Perplexity is how many tokens the model is, in effect, choosing between at each step.' }))), h('div', { class: 'table-box' }, table)),
      h('figure', { class: 'card wide' }, h('div', { class: 'card-head' }, h('div', {}, h('h3', { text: 'What each token attends to' }), h('p', { class: 'sub', text: 'Rows are the token being computed, columns the earlier tokens it reads from. The upper triangle is empty: a token cannot see the future.' })), pick), heat),
      h('figure', { class: 'card wide' }, h('div', { class: 'card-head' }, h('div', {}, h('h3', { text: 'Sampling settings' }), h('p', { class: 'sub', text: 'The same prompt and model under different decoding settings. Distinct-2 is the share of token pairs that are not repeats, averaged over five samples.' }))),
        D.samples.map(x => h('div', { class: 'sample' }, h('b', { text: `${x.setting} · distinct-2 ${x.distinct_2.toFixed(2)}` }), x.text)))));
  retokenize(); drawHeat(); lossChart(chart, D.curve);
  addEventListener('resize', () => lossChart(chart, D.curve));
  const btn = document.getElementById('theme');
  const dark = () => (document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')) === 'dark';
  btn.textContent = dark() ? 'Light mode' : 'Dark mode';
  btn.addEventListener('click', () => { document.documentElement.dataset.theme = dark() ? 'light' : 'dark'; btn.textContent = dark() ? 'Light mode' : 'Dark mode'; });
})();
