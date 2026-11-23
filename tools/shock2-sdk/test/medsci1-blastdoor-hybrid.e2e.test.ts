import assert from "node:assert/strict";
import { test } from "node:test";

import { GameServer } from "../src/index.js";

// End-to-end: the first medsci1 hybrid, behind the blast door by the
// elevator, attacks as soon as it comes through to the player.
//
// It used to step through, stop a stride from the player and stand still
// for ~2 s. Chase starts with a fresh path follower whose route is still
// in flight, so the first frame falls through to whisker avoidance - and
// the main whisker landed on the PLAYER, aiming the creature along the hit
// normal, straight back past itself. The 180 degree error stopped the body
// and played the full ogstu180 turn-around clip in the player's face.
//
// Opt-in (needs Data/ assets + compiles the runtime): npm run test:e2e
const e2eEnabled = process.env.SHOCK2_E2E === "1";

// West of the blast door, a few strides back - where the player stands
// when the hybrid comes through.
const PLAYER_AT = { x: 3.0, y: -0.36, z: -34.4 };
const DOOR_AT: [number, number, number] = [6.76, 0.4, -34.41];
const SAMPLE_FRAMES = 4;
// 6 s after the door opens - the hybrid reaches the player in ~1.5 s.
const SAMPLES = 90;
// From the first sampled Chase frame to the swing. The fresh route lands
// within a few frames; the turn-around clip held it for ~1.9 s.
const MAX_CHASE_TO_ATTACK_SECONDS = 0.5;

function distXZ(
  a: [number, number, number],
  b: [number, number, number],
): number {
  return Math.hypot(a[0] - b[0], a[2] - b[2]);
}

function prop(
  detail: { properties: { name: string; value: string }[] },
  name: string,
): string | undefined {
  return detail.properties.find((p) => p.name === name)?.value;
}

test(
  "the medsci1 blast-door hybrid attacks without turning its back",
  { skip: !e2eEnabled, timeout: 300_000 },
  async () => {
    await using game = await GameServer.launch({ mission: "medsci1.mis" });
    await game.step({ frames: 30 });

    // Entity ids differ every launch: find both by name and place
    const nearest = async (filter: string) => {
      const list = await game.entities.list({ filter, limit: 200 });
      const sorted = [...list.entities].sort(
        (a, b) => distXZ(a.position, DOOR_AT) - distXZ(b.position, DOOR_AT),
      );
      assert.ok(sorted.length > 0, `no ${filter} in medsci1`);
      return sorted[0];
    };
    const door = await nearest("Security Door");
    assert.ok(
      distXZ(door.position, DOOR_AT) < 0.5,
      `blast door not where expected: ${JSON.stringify(door.position)}`,
    );

    // The hybrid starts further east and walks up to the far side of the
    // door in a scripted sequence once the player is about.
    await game.player.teleport(PLAYER_AT);
    let hybrid = await nearest("OG-Pipe");
    for (let s = 0; s < 60 && distXZ(hybrid.position, DOOR_AT) >= 1.5; s++) {
      await game.step({ frames: 30 });
      hybrid = await nearest("OG-Pipe");
    }
    assert.ok(
      distXZ(hybrid.position, DOOR_AT) < 1.5,
      `no hybrid came up behind the blast door: ${JSON.stringify(hybrid.position)}`,
    );
    // Let it settle against the shut door, as it does in play
    await game.step({ frames: 120 });
    await game.entities.sendMessage(door.id, { type: "TurnOn" });

    let chaseAt: number | null = null;
    let attackAt: number | null = null;
    const chaseClips = new Set<string>();
    for (let i = 0; i < SAMPLES && attackAt === null; i++) {
      await game.step({ frames: SAMPLE_FRAMES });
      const detail = await game.entities.detail(hybrid.id);
      const behavior = prop(detail, "AIBehavior");
      if (behavior === "Chase") {
        chaseAt ??= i;
        const clip = (await game.entities.animation(hybrid.id))?.clip;
        if (clip) chaseClips.add(clip);
      } else if (behavior === "MeleeAttack" && chaseAt !== null) {
        attackAt = i;
      }
    }

    assert.ok(chaseAt !== null, "the hybrid never chased the player");
    assert.ok(
      attackAt !== null,
      `the hybrid chased but never swung (clips: ${[...chaseClips]})`,
    );
    const seconds = ((attackAt - chaseAt) * SAMPLE_FRAMES) / 60;
    assert.ok(
      seconds <= MAX_CHASE_TO_ATTACK_SECONDS,
      `chase to swing took ${seconds.toFixed(2)} s (clips: ${[...chaseClips]})`,
    );
  },
);
