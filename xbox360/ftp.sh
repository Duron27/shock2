#!/bin/bash
# The console's Hdd1\Games\Shock2 over Aurora's FTP, the only folder these scripts touch:
#   ftp.sh push hello|triangle   send <title>/build/rust_<title>.xex
#   ftp.sh log hello|triangle    print rust_<title>.log
#   ftp.sh ls                    list the folder
# XENON_FTP (default 192.168.1.65) and XENON_FTP_LOGIN (default Aurora's xboxftp:xboxftp) override the console's.
# Aurora's FTP is gone while a title runs: go back to the dashboard first.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
host="${XENON_FTP:-192.168.1.65}"
login="${XENON_FTP_LOGIN:-xboxftp:xboxftp}"
remote="ftp://$host/Hdd1/Games/Shock2"
case "${1:-}" in
push)
	xex="$HERE/${2:?title}/build/rust_$2.xex"
	curl -sS --connect-timeout 5 --ftp-create-dirs --user "$login" -T "$xex" "$remote/"
	echo "sent $(basename "$xex") ($(md5sum <"$xex" | cut -c1-8)) to Hdd1\\Games\\Shock2"
	;;
log) curl -sS --connect-timeout 5 --user "$login" "$remote/rust_${2:?title}.log" ;;
ls) curl -sS --connect-timeout 5 --user "$login" "$remote/" ;;
*) sed -n '2,6p' "$0"; exit 1 ;;
esac
