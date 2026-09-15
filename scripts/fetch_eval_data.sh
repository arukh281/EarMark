#!/usr/bin/env bash
# Download the small public evaluation assets into ./.cache and verify their pinned SHA-256.
#
#     scripts/fetch_eval_data.sh [--list] [--verify-only] [GROUP ...]
#
# Groups (default: all of them):
#   vbd    VoiceBank+DEMAND test set (824 utterances, 48 kHz; resampled to 16 kHz at load time)
#          from the Edinburgh DataShare (handle 10283/2791, CC BY 4.0), about 326 MB of zips.
#   gtcrn  official GTCRN checkpoints (VoiceBank+DEMAND and DNS3), MIT licence, about 1.3 MB.
#
# URLs, sizes and SHA-256 pins live in python/earmark/eval/assets.tsv, which the Python
# loaders read too, so there is exactly one place to change a pin. Downloads land in
# $EARMARK_CACHE (default: <repo>/.cache, which git ignores). A file whose size or hash does
# not match its pin is deleted and the script fails; nothing unverified is ever extracted.
# Existing verified files are not downloaded again, so re-running the script is cheap.
#
# Exit status: 0 success, 1 download or verification failure, 2 usage error.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd -P)"
manifest="$repo_root/python/earmark/eval/assets.tsv"
cache="${EARMARK_CACHE:-$repo_root/.cache}"

list_only=0
verify_only=0
groups=""

usage() {
  sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --list) list_only=1 ;;
    --verify-only) verify_only=1 ;;
    -h | --help)
      usage
      exit 0
      ;;
    -*)
      echo "fetch_eval_data: unknown option $1" >&2
      usage >&2
      exit 2
      ;;
    *) groups="$groups $1" ;;
  esac
  shift
done

if [ ! -f "$manifest" ]; then
  echo "fetch_eval_data: asset manifest not found: $manifest" >&2
  exit 2
fi

known_groups="$(grep -v '^#' "$manifest" | awk -F '\t' 'NF >= 7 { print $2 }' | sort -u | tr '\n' ' ')"
for g in $groups; do
  case " $known_groups " in
    *" $g "*) ;;
    *)
      echo "fetch_eval_data: unknown group '$g' (known: $known_groups)" >&2
      exit 2
      ;;
  esac
done
[ -n "$groups" ] || groups="$known_groups"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{ print $1 }'
  else
    shasum -a 256 "$1" | awk '{ print $1 }'
  fi
}

size_of() {
  wc -c <"$1" | tr -d ' '
}

# Succeeds when FILE exists with the pinned size and SHA-256.
is_verified() {
  local file="$1" bytes="$2" sha="$3"
  [ -f "$file" ] || return 1
  [ "$(size_of "$file")" = "$bytes" ] || return 1
  [ "$(sha256_of "$file")" = "$sha" ] || return 1
}

fetch_one() {
  local name="$1" bytes="$2" sha="$3" dest="$4" url="$5"
  if is_verified "$dest" "$bytes" "$sha"; then
    echo "fetch_eval_data: ok       $name (sha256 verified)"
    return 0
  fi
  if [ "$verify_only" -eq 1 ]; then
    echo "fetch_eval_data: MISSING  $name (expected at $dest)" >&2
    return 1
  fi
  mkdir -p "$(dirname "$dest")"
  local part="$dest.part"
  rm -f "$part"
  echo "fetch_eval_data: download $name ($bytes bytes)"
  if ! curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 -sS -o "$part" "$url"; then
    rm -f "$part"
    echo "fetch_eval_data: download failed for $name from $url" >&2
    return 1
  fi
  local got_bytes got_sha
  got_bytes="$(size_of "$part")"
  got_sha="$(sha256_of "$part")"
  if [ "$got_bytes" != "$bytes" ] || [ "$got_sha" != "$sha" ]; then
    rm -f "$part"
    echo "fetch_eval_data: VERIFY FAILED for $name" >&2
    echo "  expected $bytes bytes, sha256 $sha" >&2
    echo "  got      $got_bytes bytes, sha256 $got_sha" >&2
    return 1
  fi
  mv "$part" "$dest"
  echo "fetch_eval_data: ok       $name (downloaded, sha256 verified)"
}

extract_one() {
  local name="$1" sha="$2" archive="$3" into="$4"
  local stamp="$into/.extracted_$name.sha256"
  if [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$sha" ]; then
    echo "fetch_eval_data: ok       $name already extracted into $into"
    return 0
  fi
  if [ "$verify_only" -eq 1 ]; then
    echo "fetch_eval_data: NOT EXTRACTED $name (run without --verify-only)" >&2
    return 1
  fi
  mkdir -p "$into"
  unzip -q -o "$archive" -d "$into"
  rm -rf "$into/__MACOSX" # macOS resource forks shipped inside some of the zips
  printf '%s\n' "$sha" >"$stamp"
  echo "fetch_eval_data: ok       $name extracted into $into"
}

count_wavs() {
  find "$1" -maxdepth 1 -type f -name '*.wav' 2>/dev/null | wc -l | tr -d ' '
}

check_vbd() {
  local root="$cache/data/vbd" status=0 n sub
  for sub in clean_testset_wav noisy_testset_wav; do
    n="$(count_wavs "$root/$sub")"
    if [ "$n" != "824" ]; then
      echo "fetch_eval_data: expected 824 wavs in $root/$sub, found $n" >&2
      status=1
    fi
  done
  if [ ! -f "$root/logfiles/log_testset.txt" ]; then
    echo "fetch_eval_data: missing $root/logfiles/log_testset.txt" >&2
    status=1
  fi
  [ "$status" -eq 0 ] && echo "fetch_eval_data: ok       vbd layout (824 clean + 824 noisy wavs, noise log)"
  return "$status"
}

if [ "$list_only" -eq 1 ]; then
  grep -v '^#' "$manifest" | awk -F '\t' 'NF >= 7 { printf "%-28s %-6s %11s  %s\n", $1, $2, $3, $4 }'
  exit 0
fi

if [ "$verify_only" -eq 0 ]; then
  "$repo_root/scripts/disk_guard.sh" "$repo_root"
fi

failed=0
for group in $groups; do
  while IFS="$(printf '\t')" read -r name grp bytes sha dest extract url; do
    case "$name" in '' | '#'*) continue ;; esac
    [ "$grp" = "$group" ] || continue
    if ! fetch_one "$name" "$bytes" "$sha" "$cache/$dest" "$url"; then
      failed=1
      continue
    fi
    if [ "$extract" != "-" ]; then
      extract_one "$name" "$sha" "$cache/$dest" "$cache/$extract" || failed=1
    fi
  done <"$manifest"
  if [ "$group" = "vbd" ] && [ "$failed" -eq 0 ]; then
    check_vbd || failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "fetch_eval_data: FAILED (see messages above)" >&2
  exit 1
fi
echo "fetch_eval_data: all requested assets verified under $cache"
