#!/usr/bin/env bash
#bash script that converts the .dat and .grd TCAD files into .apf field files
set -euo pipefail

ALLPIX_DIR="/home/claire/allpix-squared"
MESH_DIR="/mnt/c/Users/clair/OneDrive/Documents/GEARE 397/Prague/CAPADS/tcad_files"  # where TCAD .grd/.dat live
OUT_DIR="${ALLPIX_DIR}/Neutrons/Meshes"       # output APF destination
CONF="${ALLPIX_DIR}/Neutrons/tmp/mesh_tcad.conf"
TMP_DIR="${MESH_DIR}/.mesh_converter_tmp"

mkdir -p "${OUT_DIR}" "${TMP_DIR}"

# Expect exactly one .grd and many .dat files
GRD_FILE="$(ls -1 "${MESH_DIR}"/*.grd | head -n 1)"
if [[ -z "${GRD_FILE}" ]]; then
  echo "No .grd file found in ${MESH_DIR}"
  exit 1
fi

for dat in "${MESH_DIR}"/*.dat; do
  [[ -f "${dat}" ]] || continue
  base="$(basename "${dat}" .dat)"

  # Create temp prefix with shared .grd + per-case .dat
  ln -sf "${GRD_FILE}" "${TMP_DIR}/${base}.grd"
  ln -sf "${dat}" "${TMP_DIR}/${base}.dat"

  # mesh_converter writes <output_prefix>_<observable>_interpolated.apf
  /home/claire/allpix-squared/bin/mesh_converter -f "${TMP_DIR}/${base}" -c "${CONF}" -o "${OUT_DIR}/${base}"
done
