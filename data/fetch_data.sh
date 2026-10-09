#!/usr/bin/env bash
#
# Fetch the synthetic 70S particle stacks used by Sections 6.4 and 6.5.
# Everything else in this repository runs without external data.
#
#   ./data/fetch_data.sh 70s      synthetic cryo-EM stacks, 1.9 GB, from this
#                                 repository's datasets-v1 release (needs gh)
#
# Each archive is checked against the SHA-256 recorded when the study ran.  A
# mismatch stops the script: a silently different input would invalidate every
# number the run reports.
set -Eeuo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
seventy_s_root="${repo_root}/data/70s"


declare -A SEVENTY_S_SHA=(
  [70s_white_on_black_snr004_seed0.tar]=2a4be66fe8951bac38aefe1f2e05541afc319337300659bc49554495156c7ee8
  [70s_white_on_black_snr005_seed0.tar]=8fb7a912cd22ea5e5de2100723c04971926ac443881393ee1650bda8ebbaeb3f
  [70s_white_on_black_snr006_seed0.tar]=e820289b75e4fc534e58a012343f2706c153a09ad6a069b4a86cc7b65d10529b
)

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "$1 is required but was not found on PATH." >&2; exit 127; }
}

check() {  # check <file> <expected sha256>
  local actual
  actual="$(sha256sum -- "$1" | cut -d' ' -f1)"
  if [ "${actual}" != "$2" ]; then
    echo "SHA-256 mismatch for $1" >&2
    echo "  expected ${2}" >&2
    echo "  actual   ${actual}" >&2
    exit 1
  fi
  echo "  sha256 ok"
}

fetch_70s() {
  need gh
  mkdir -p -- "${seventy_s_root}"
  echo "70S cells -> ${seventy_s_root}"
  for name in "${!SEVENTY_S_SHA[@]}"; do
    local archive="${seventy_s_root}/${name}"
    if [ ! -f "${archive}" ]; then
      echo "  downloading ${name} (647 MB)"
      gh release download datasets-v1 --repo phonchi/fourier-2sdr-clean \
        --pattern "${name}" --dir "${seventy_s_root}"
    fi
    check "${archive}" "${SEVENTY_S_SHA[$name]}"
    tar -xf "${archive}" -C "${seventy_s_root}"
    rm -f -- "${archive}"
  done
  echo "70S ready."
}

case "${1:-}" in
  70s)     fetch_70s ;;
  *)       sed -n '3,7p' -- "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
