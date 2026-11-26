#!/bin/bash
# The console's Hdd1\Games\Shock2 over Aurora's FTP, the only folder these scripts touch:
#   ftp.sh push hello|triangle   send <title>/build/rust_<title>.xex
#   ftp.sh log hello|triangle    print rust_<title>.log
#   ftp.sh ls                    list the folder
#   ftp.sh data [dir]            copy the game's data (default ../Data, symlinks followed) to Hdd1\Games\Shock2\Data
# XENON_FTP (default 192.168.1.65) and XENON_FTP_LOGIN (default Aurora's xboxftp:xboxftp) override the console's.
# Aurora's FTP is gone while a title runs: go back to the dashboard first.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
host="${XENON_FTP:-192.168.1.65}"
login="${XENON_FTP_LOGIN:-xboxftp:xboxftp}"
remote="ftp://$host/Hdd1/Games/Shock2"
case "${1:-}" in
push)
	xex=$(ls "$HERE/${2:?title}"/build/*.xex | head -1)
	curl -sS --connect-timeout 5 --ftp-create-dirs --user "$login" -T "$xex" "$remote/"
	echo "sent $(basename "$xex") ($(md5sum <"$xex" | cut -c1-8)) to Hdd1\\Games\\Shock2"
	;;
log)
	name="rust_${2:?title}"; [ "$2" = datacheck ] && name=xbox360_datacheck
	curl -sS --connect-timeout 5 --user "$login" "$remote/$name.log"
	;;
data)
	# Resumable: a file already there at its size is skipped, and one that fails is tried again after a pause.
	src="$(cd "${2:-$HERE/../Data}" && pwd -P)"
	cd "$src"
	total=$(find -L . -type f | wc -l); n=0; failed=(); skipped=()
	while IFS= read -r -d '' f; do
		n=$((n + 1)); rel="${f#./}"; size=$(stat -L -c %s "$f")
		# FATX holds names of at most 42 characters, so the console refuses longer ones (550).
		if [ "$(basename "$rel" | wc -c)" -gt 43 ]; then echo "[$n/$total] $rel: name too long for FATX, skipped"; skipped+=("$rel"); continue; fi
		there=$(curl -sS --connect-timeout 5 --user "$login" -I "$remote/Data/$rel" 2>/dev/null | tr -d '\r' | awk '/^Content-Length:/ {print $2}' || true)
		if [ "$there" = "$size" ]; then echo "[$n/$total] $rel: already there"; continue; fi
		ok=
		for attempt in 1 2 3 4 5; do
			if curl -sS --connect-timeout 10 --speed-time 30 --speed-limit 1024 --ftp-create-dirs --user "$login" \
				-T "$f" "$remote/Data/$rel"; then ok=1; break; fi
			echo "[$n/$total] $rel: attempt $attempt failed, waiting 10 s"; sleep 10
		done
		if [ -n "$ok" ]; then echo "[$n/$total] $rel ($size bytes)"; else failed+=("$rel"); fi
	done < <(find -L . -type f -print0 | sort -z)
	if [ ${#failed[@]} -gt 0 ]; then printf 'not copied: %s\n' "${failed[@]}"; exit 1; fi
	echo "$((total - ${#skipped[@]})) of $total files on the console; ${#skipped[@]} skipped for their names"
	;;
ls) curl -sS --connect-timeout 5 --user "$login" "$remote/" ;;
*) sed -n '2,7p' "$0"; exit 1 ;;
esac
