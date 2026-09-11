#!/usr/bin/env bash
# Fail when the volume holding PATH (default: the repository root) has less free space
# than EARMARK_MIN_FREE_GB (default 5). Always prints the free space.
#
#     scripts/disk_guard.sh [PATH]
#
# Exit status: 0 enough space, 1 below the threshold, 2 usage error.
# Sizes are GiB (1024^3 bytes), which is what `df -g` reports on macOS.
set -euo pipefail

min_gb="${EARMARK_MIN_FREE_GB:-5}"
target="${1:-$(cd "$(dirname "$0")/.." && pwd -P)}"

if [ ! -e "$target" ]; then
  echo "disk_guard: no such path: $target" >&2
  exit 2
fi
case "$min_gb" in
  '' | *[!0-9.]*)
    echo "disk_guard: EARMARK_MIN_FREE_GB must be a number, got '$min_gb'" >&2
    exit 2
    ;;
esac

# POSIX output: 1024-byte blocks, one line per filesystem; column 4 is "Available".
avail_kb="$(df -Pk "$target" | awk 'NR == 2 { print $4 }')"
mount_point="$(df -Pk "$target" | awk 'NR == 2 { print $NF }')"
if [ -z "$avail_kb" ]; then
  echo "disk_guard: could not read free space for $target" >&2
  exit 2
fi
free_gb="$(awk -v kb="$avail_kb" 'BEGIN { printf "%.1f", kb / 1048576 }')"

if awk -v kb="$avail_kb" -v min="$min_gb" 'BEGIN { exit !(kb / 1048576 >= min) }'; then
  echo "disk_guard: OK, ${free_gb} GiB free on ${mount_point} (minimum ${min_gb} GiB)"
  exit 0
fi
echo "disk_guard: LOW DISK, only ${free_gb} GiB free on ${mount_point} (minimum ${min_gb} GiB)" >&2
echo "disk_guard: free space first, e.g. make clean-caches" >&2
exit 1
