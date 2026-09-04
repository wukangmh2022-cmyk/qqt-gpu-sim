#!/usr/bin/env node
/* Verify bush hide and recovery semantics in sim and render logic */
'use strict';

const fs = require('fs');
const path = require('path');
const Q = require(path.join(__dirname, '..', 'web', 'sim.js'));

const levels = JSON.parse(fs.readFileSync(
  path.join(__dirname, '..', 'web', 'assets', 'maps', 'levels.json'), 'utf8'));

// 1. Find a map with bush (野外01, id=28)
const levelField = levels.find((l) => l.name === '野外01' || l.id === 28);
if (!levelField) throw new Error('Could not find 野外01 map');

// Find a map with overhead tent/cottage (沙漠01, id=2)
const levelDesert = levels.find((l) => l.name === '沙漠01' || l.id === 2);
if (!levelDesert) throw new Error('Could not find 沙漠01 map');

// Render logic mirror from web/main.js:
function createRenderChecker(sim) {
  const N = sim.N || (13 * 15);
  const W = sim.W || 15;
  const H = sim.H || 13;

  const hideCells = new Set();
  const isLevelBush = (idx) => sim.level && (
    (sim.level.bush && sim.level.bush[idx]) ||
    (sim.level.layers && Math.abs(sim.level.layers[0][idx] || 0) === 6003)
  );

  if (sim.level) {
    if (sim.level.overhead) {
      for (let i = 0; i < N; i++) {
        if (sim.level.overhead[i] && !isLevelBush(i)) hideCells.add(i);
      }
    }
    if (sim.level.cover) {
      for (let i = 0; i < N; i++) {
        if (sim.level.cover[i] && !isLevelBush(i)) hideCells.add(i);
      }
    }
    if (sim.cover) {
      for (let i = 0; i < N; i++) {
        if (sim.cover[i] && !isLevelBush(i)) hideCells.add(i);
      }
    }
  }

  // StructList for living bushes
  if (sim.level && sim.level.bush) {
    for (let i = 0; i < N; i++) {
      if (sim.bush && sim.bush[i]) {
        hideCells.add(i);
      }
    }
  }

  function isCellCovered(cellIdx) {
    if (cellIdx < 0 || cellIdx >= N) return false;
    if (isLevelBush(cellIdx) && (!sim.bush || !sim.bush[cellIdx])) return false;
    if (hideCells.has(cellIdx)) return true;
    if (sim.cover && sim.cover[cellIdx] && !isLevelBush(cellIdx)) return true;
    if (sim.level && sim.level.overhead && sim.level.overhead[cellIdx] && !isLevelBush(cellIdx)) return true;
    return false;
  }

  return { isCellCovered, hideCells };
}

// TEST 1: Desert permanent overhead cottage
{
  const sim = new Q.Sim(42);
  sim.reset(levelDesert);
  const overheadIdx = levelDesert.overhead.findIndex((v) => v === 1);
  if (overheadIdx < 0) throw new Error('Desert map must have overhead');
  
  const rc = createRenderChecker(sim);
  if (!rc.isCellCovered(overheadIdx)) {
    throw new Error(`Permanent overhead at cell ${overheadIdx} should be covered!`);
  }
  console.log('✓ Permanent overhead tent correctly covered');
}

// TEST 2: Bush lifecycle in 野外01
{
  const sim = new Q.Sim(42);
  sim.reset(levelField);

  // Bush indices
  const bushIndices = [];
  for (let i = 0; i < sim.bush.length; i++) {
    if (sim.bush[i]) bushIndices.push(i);
  }
  if (bushIndices.length === 0) throw new Error('野外01 must have bush cells');
  console.log(`Found ${bushIndices.length} bush cells in 野外01`);

  // Verify none of them are marked as cover or overhead in levels.json
  for (const b of bushIndices) {
    if (levelField.overhead && levelField.overhead[b]) {
      throw new Error(`Bush cell ${b} has overhead=1 in level data!`);
    }
    if (levelField.cover && levelField.cover[b]) {
      throw new Error(`Bush cell ${b} has cover=1 in level data!`);
    }
    if (sim.cover[b] !== 0) {
      throw new Error(`Bush cell ${b} has sim.cover=${sim.cover[b]}!`);
    }
  }
  console.log('✓ All bush cells have cover=0 and overhead=0');

  // Verify that when bush is alive, isCellCovered returns true
  const testBushCell = bushIndices[0];
  let rc = createRenderChecker(sim);
  if (!rc.isCellCovered(testBushCell)) {
    throw new Error(`Living bush at cell ${testBushCell} should be covered!`);
  }
  console.log(`✓ Living bush cell ${testBushCell} is correctly covered`);

  // Place player 0 directly in the bush cell
  const r = Math.floor(testBushCell / sim.W);
  const c = testBushCell % sim.W;
  sim.pos[0] = r + 0.5;
  sim.pos[1] = c + 0.5;

  // Place bomb in bush cell
  sim.fuse[testBushCell] = 1; // trigger explosion next step
  sim.bombBlast[testBushCell] = 2;
  sim.owner[testBushCell] = 0;

  // Step to trigger explosion
  sim.step([[Q.MOVE_IDLE, 0], [Q.MOVE_IDLE, 0]]);

  // Verify bush was destroyed
  if (sim.bush[testBushCell] !== 0) {
    throw new Error(`Bush at cell ${testBushCell} should be destroyed by explosion!`);
  }
  if (sim.cover[testBushCell] !== 0) {
    throw new Error(`Bush at cell ${testBushCell} must have sim.cover=0 after explosion!`);
  }
  console.log(`✓ Bush cell ${testBushCell} was destroyed by explosion`);

  // Check render checker after destruction
  rc = createRenderChecker(sim);
  if (rc.isCellCovered(testBushCell)) {
    throw new Error(`Destroyed bush cell ${testBushCell} should NOT be covered! Visibility must be restored!`);
  }
  console.log(`✓ Destroyed bush cell ${testBushCell} is NO LONGER covered (visible)`);

  // Move player around and come back to testBushCell
  sim.pos[0] = r + 0.5;
  sim.pos[1] = c + 0.5;
  const pcell = Math.floor(sim.pos[0]) * sim.W + Math.floor(sim.pos[1]);
  if (rc.isCellCovered(pcell)) {
    throw new Error(`Player standing in destroyed bush cell ${pcell} must NOT be hidden!`);
  }
  console.log(`✓ Player standing in destroyed bush cell is fully VISIBLE`);
}

console.log('ALL BUSH VISIBILITY TESTS PASSED!');
