import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { GameServer } from "../src/index.js";
import { clickCanvas as click, pauseEntry } from "./helpers/frontend-menu.js";

const enabled = process.env.SHOCK2_E2E === "1";
const DONE: [number, number] = [320, 436];
const OPTIONS_ENTRY: [number, number] = [490, 202];
/** The three live tab slots (`options_panel::TABS`): Controls, Display, Audio. */
const tab = (index: number): [number, number] => [80 + index * 160, 36];
/** A settings row: 29px apart from the first row's centre. */
const row = (index: number): [number, number] => [333, 84 + index * 29];

type PcSettings = {
  mouse_sensitivity: number;
  invert_mouse_y: boolean;
  field_of_view: number;
  brightness: number;
  music_volume: number;
};

// The flatscreen-only branch's Options pages are PC settings (mouse, display,
// audio); the VR comfort pages, and their vignette and grid, are gone.
test("Options saves PC settings from the main and pause menus", {
  skip: !enabled && "set SHOCK2_E2E=1 to run",
  timeout: 300_000,
}, async () => {
  const root = await mkdtemp(join(tmpdir(), "shock2-options-test-"));
  const path = join(root, "settings.json");
  const previous = process.env.SHOCK2_SETTINGS_PATH;
  process.env.SHOCK2_SETTINGS_PATH = path;
  const saved = async () => (JSON.parse(await readFile(path, "utf8")) as { pc: PcSettings }).pc;
  try {
    {
      await using game = await GameServer.launch({ mission: "main_menu" });
      await game.step({ frames: 10 });
      await click(game, OPTIONS_ENTRY);
      await click(game, row(0)); // Mouse sensitivity 1.0 -> 1.25
      assert.equal((await saved()).mouse_sensitivity, 1.25);
      await click(game, row(1)); // Invert mouse Off -> On
      assert.equal((await saved()).invert_mouse_y, true);
      await click(game, tab(1));
      await click(game, row(0)); // Field of view 45 -> 50
      await click(game, row(1)); // Brightness 1.4 -> 1.6
      assert.equal((await saved()).field_of_view, 50);
      assert.ok(Math.abs((await saved()).brightness - 1.6) < 1e-4);
      await click(game, tab(2));
      await click(game, row(1)); // Music volume 100% wraps to 0%
      assert.equal((await saved()).music_volume, 0);
      await click(game, DONE);
      assert.equal((await game.info()).mission, "main_menu");
    }
    {
      await using game = await GameServer.launch({ mission: "debug_minimal" });
      await game.step({ frames: 10 });
      await game.input.trigger("TogglePauseMenu");
      await game.step({ frames: 3 });
      await click(game, pauseEntry(3));
      // This process must start from the previous process's saved values.
      await click(game, row(0)); // Mouse sensitivity 1.25 -> 1.5
      assert.equal((await saved()).mouse_sensitivity, 1.5);
      assert.equal((await saved()).invert_mouse_y, true);
      await click(game, tab(1));
      await click(game, row(4)); // Reset display defaults; controls untouched
      const after = await saved();
      assert.equal(after.field_of_view, 45);
      assert.ok(Math.abs(after.brightness - 1.4) < 1e-4);
      assert.equal(after.mouse_sensitivity, 1.5);
      assert.equal(after.music_volume, 0);
      await click(game, DONE);
      assert.equal((await game.info()).paused, true);
      await click(game, pauseEntry(0));
      assert.equal((await game.info()).paused, false);
      assert.equal((await game.info()).mission, "debug_minimal");
    }
  } finally {
    if (previous === undefined) delete process.env.SHOCK2_SETTINGS_PATH;
    else process.env.SHOCK2_SETTINGS_PATH = previous;
    await rm(root, { recursive: true, force: true });
  }
});
