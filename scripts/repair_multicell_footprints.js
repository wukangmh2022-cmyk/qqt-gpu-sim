#!/usr/bin/env node
/* Restore collision cells covered by multi-cell layer1 structures. */
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const levelsPath = path.join(ROOT, 'web', 'assets', 'maps', 'levels.json');
const elementsPath = path.join(ROOT, 'web', 'assets', 'maps', 'elements.json');
const write = process.argv.includes('--write');

const levels = JSON.parse(fs.readFileSync(levelsPath, 'utf8'));
const elements = JSON.parse(fs.readFileSync(elementsPath, 'utf8'));

let structures = 0;
let changed = 0;
let skipped = 0;
const examples = [];

for (const level of levels) {
  const w = level.w;
  const h = level.h;
  const layer = level.layers && level.layers[1];
  if (!layer) continue;

  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const raw = layer[r * w + c];
      if (!raw || raw < 0) continue;
      const eid = Math.abs(raw);
      const element = elements[String(eid)];
      if (!element || (element.w <= 1 && element.h <= 1)) continue;

      const origin = r * w + c;
      const isWall = !!level.wall[origin];
      const isBrick = !!level.brick[origin];
      if (!isWall && !isBrick) {
        skipped++;
        continue;
      }
      structures++;

      for (let dr = 0; dr < element.h; dr++) {
        for (let dc = 0; dc < element.w; dc++) {
          const rr = r + dr;
          const cc = c + dc;
          if (rr >= h || cc >= w) continue;
          const i = rr * w + cc;
          if (isWall) {
            if (!level.wall[i]) {
              level.wall[i] = 1;
              changed++;
              if (examples.length < 8) examples.push(`${level.source} (${rr},${cc}) wall`);
            }
          } else if (!level.brick[i]) {
            level.brick[i] = 1;
            changed++;
            if (examples.length < 8) examples.push(`${level.source} (${rr},${cc}) brick`);
          }
        }
      }
    }
  }
}

console.log(JSON.stringify({ write, structures, skipped, changed, examples }, null, 2));

if (write && changed > 0) {
  fs.writeFileSync(levelsPath, JSON.stringify(levels));
}
