import assert from "node:assert/strict";
import { test } from "node:test";

import { GameServer } from "../src/index.js";

// Space (the jump key) or Escape skips the movie at once - no hold, nothing on
// screen. Booting the runtime straight into a cutscene gives it `ShowMainMenu`
// as its follow-on, so the menu appearing is the observable skip.
const e2eEnabled = process.env.SHOCK2_E2E === "1";

/** The intro movie, ~2:40 long - far longer than anything stepped here. */
const INTRO = "cs1.avi";

test(
  "pressing Space skips a cutscene to its follow-on at once",
  { skip: !e2eEnabled && "set SHOCK2_E2E=1 to run", timeout: 300_000 },
  async () => {
    await using game = await GameServer.launch({ mission: INTRO });
    await game.step({ frames: 10 });
    assert.equal((await game.info()).mission, INTRO, "the movie should be playing");

    await game.input.set("jump", 1.0);
    await game.step({ frames: 2 });
    await game.input.set("jump", 0.0);
    await game.step({ frames: 2 });

    assert.equal((await game.info()).mission, "main_menu", "one press should end the movie");
  },
);

test(
  "Escape skips a cutscene, and a movie left alone keeps playing",
  { skip: !e2eEnabled && "set SHOCK2_E2E=1 to run", timeout: 300_000 },
  async () => {
    await using game = await GameServer.launch({ mission: INTRO });
    await game.step({ frames: 300 });
    assert.equal((await game.info()).mission, INTRO, "an untouched movie should still be playing");

    await game.input.trigger("TogglePauseMenu");
    await game.step({ frames: 2 });
    assert.equal((await game.info()).mission, "main_menu", "Escape should end the movie");
  },
);
