#!/bin/bash
# shader_sweep.sh <dump dir> [scene...]: runs the debug runtime over scenes with SHOCK2_SHADER_DUMP set, so the engine
# writes every shader it builds, for the console as well (xbox360/README.md, "Shaders"). Each scene steps, opens the
# use mode, map, pause menu and a log, spawns a monster and equips and fires each weapon and psi, so the programs
# those draw with get made. SWEEP_WINDOW adds window flags (say "--visible --window-pos 1960,60"); hidden by default.
set -u
DUMP="$(realpath -m "$1")"; shift
SCENES=("$@")
[ ${#SCENES[@]} -eq 0 ] && SCENES=(menu debug_weapons debug_psi debug_ai_ranged earth.mis medsci1.mis)
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/target/debug/debug_runtime"
mkdir -p "$DUMP"
cd "$ROOT"
post() { # post <path> <json>: fails loudly on an HTTP error or a rejected channel
	local out
	out=$(curl -s -w '\n%{http_code}' -X POST "$u$1" -d "$2")
	case "${out##*$'\n'}" in 2*) ;; *) echo "  $1 $2 -> ${out%$'\n'*}" ;; esac
	case "$out" in *rejected*|*error*) echo "  $1 $2 -> ${out%$'\n'*}" ;; esac
}
for scene in "${SCENES[@]}"; do
	log=$(mktemp)
	args=(--port 0 ${SWEEP_WINDOW:-})
	[ "$scene" != menu ] && args+=(--mission "$scene")
	# Silent: ALSA's null device (silent_alsa.conf), as the sweep fires every weapon.
	ALSA_CONFIG_PATH="${SWEEP_ALSA_CONFIG-$ROOT/xbox360/tools/silent_alsa.conf}" \
		SHOCK2_SHADER_DUMP="$DUMP" "$BIN" "${args[@]}" > "$log" 2>&1 &
	pid=$!
	port=
	for _ in $(seq 1 600); do
		port=$(grep -oE 'SHOCK2QUEST_PORT port=[0-9]+' "$log" | head -1 | cut -d= -f2)
		[ -n "$port" ] && curl -s "http://127.0.0.1:$port/v1/health" > /dev/null 2>&1 && break
		kill -0 $pid 2>/dev/null || break
		sleep 0.5
	done
	if [ -z "$port" ]; then echo "$scene: never came up"; tail -3 "$log"; continue; fi
	u="http://127.0.0.1:$port"
	post /v1/step '{"frames":120}'
	for a in ToggleUseMode ToggleUseMode ToggleMap ToggleMap TogglePauseMenu TogglePauseMenu ReadLastUnreadLog \
		ReadLastUnreadLog SpawnDebugMonster EquipPistol FIRE EquipLaserPistol FIRE EquipShotgun FIRE EquipFusionCannon FIRE \
		EquipEmpRifle FIRE EquipGrenadeLauncher FIRE EquipStasisFieldGenerator FIRE EquipWrench FIRE EquipPsiAmp FIRE \
		CyclePsiPower FIRE DebugAlertAll; do
		if [ "$a" = FIRE ]; then
			post /v1/control/input '{"right_hand.trigger": 1.0}'
			post /v1/step '{"frames":20}'
			post /v1/control/input '{"right_hand.trigger": 0.0}'
			post /v1/step '{"frames":40}'
			continue
		fi
		post /v1/input/action "{\"action\":\"$a\"}"
		post /v1/step '{"frames":30}'
	done
	post /v1/step '{"frames":120}'
	curl -s -X POST "$u/v1/shutdown" > /dev/null
	wait $pid
	echo "$scene: $(ls "$DUMP"/xbox360/*.spv 2>/dev/null | grep -vc compiled) console modules so far"
done
