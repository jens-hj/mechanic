#!/usr/bin/env bash

set -euo pipefail

usage() {
    printf 'usage: %s [--check] ARCHIVE\n' "${0##*/}" >&2
    exit 2
}

check_only=false
if [[ ${1:-} == "--check" ]]; then
    check_only=true
    shift
fi
[[ $# -eq 1 ]] || usage

archive=$1
[[ -f $archive ]] || {
    printf 'archive not found: %s\n' "$archive" >&2
    exit 1
}

for required_command in awk cmp comm diff ffmpeg grep install mktemp sort sips unzip; do
    command -v "$required_command" >/dev/null || {
        printf 'required command not found: %s\n' "$required_command" >&2
        exit 1
    }
done

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
block_thumbnail_renderer="$script_dir/render-block-thumbnail.sh"
repo_root=$(cd "$script_dir/../../../.." && pwd -P)
asset_root="$repo_root/crates/mechanic-app/assets/materials"
[[ -d $asset_root ]] || {
    printf 'material asset directory not found: %s\n' "$asset_root" >&2
    exit 1
}

staging_dir=$(mktemp -d "${TMPDIR:-/tmp}/mechanic-material-import.XXXXXX")
cleanup() {
    rm -rf -- "$staging_dir"
}
trap cleanup EXIT

materials=(aluminium carbon carbon_fiber concrete iron plastic rubber steel stone wood)
maps=(base_color normal orm)
expected_manifest="$staging_dir/expected-manifest"
actual_manifest="$staging_dir/actual-manifest"
expected_materials="$staging_dir/expected-materials"
actual_materials="$staging_dir/actual-materials"

for material in "${materials[@]}"; do
    for map in "${maps[@]}"; do
        printf 'materials_styled/%s/%s_%s.png\n' "$material" "$material" "$map"
    done
done | LC_ALL=C sort >"$expected_manifest"

printf '%s\n' "${materials[@]}" | LC_ALL=C sort >"$expected_materials"

unzip -tqq "$archive"
unzip -Z1 "$archive" | LC_ALL=C sort >"$actual_manifest"
awk -F/ '$1 == "materials_styled" && NF >= 3 { print $2 }' "$actual_manifest" \
    | LC_ALL=C sort -u >"$actual_materials"
if ! cmp -s "$expected_materials" "$actual_materials"; then
    printf 'archive material set differs from the supported set; ask the user how to handle additions and omissions:\n' >&2
    printf 'added in archive:\n' >&2
    comm -13 "$expected_materials" "$actual_materials" >&2
    printf 'missing from archive:\n' >&2
    comm -23 "$expected_materials" "$actual_materials" >&2
    exit 1
fi

while IFS= read -r required; do
    if ! grep -Fqx "$required" "$actual_manifest"; then
        printf 'archive is missing canonical map: %s\n' "$required" >&2
        exit 1
    fi
done <"$expected_manifest"

extracted_root="$staging_dir/extracted"
output_root="$staging_dir/output"
mkdir -p "$extracted_root" "$output_root"
unzip -q "$archive" -d "$extracted_root"

image_dimensions() {
    local image=$1
    local info width height
    info=$(sips -1 -g format -g pixelWidth -g pixelHeight -g bitsPerSample -g samplesPerPixel "$image")
    [[ $info == *'|format: png|'* && $info == *'|bitsPerSample: 8|'* && $info == *'|samplesPerPixel: 4|'* ]] || {
        printf 'expected an 8-bit RGBA PNG: %s\n' "$image" >&2
        exit 1
    }
    [[ $info =~ \|pixelWidth:\ ([0-9]+)\| ]] || exit 1
    width=${BASH_REMATCH[1]}
    [[ $info =~ \|pixelHeight:\ ([0-9]+)\| ]] || exit 1
    height=${BASH_REMATCH[1]}
    printf '%s %s\n' "$width" "$height"
}

normalized_count=0
for material in "${materials[@]}"; do
    mkdir -p "$output_root/$material"
    for map in "${maps[@]}"; do
        source="$extracted_root/materials_styled/$material/${material}_${map}.png"
        output="$output_root/$material/${material}_${map}.png"
        read -r width height < <(image_dimensions "$source")
        [[ $width == "$height" && ( $width == 1024 || $width == 2048 || $width == 3072 ) ]] || {
            printf 'expected a 1024, 2048, or 3072 pixel square map: %s (%sx%s)\n' \
                "$source" "$width" "$height" >&2
            exit 1
        }
        cp "$source" "$output"
        if [[ $width != 3072 ]]; then
            sips -z 3072 3072 "$output" >/dev/null
            ((normalized_count += 1))
        fi
    done

    thumbnail="$output_root/$material/${material}_thumbnail.png"
    cp "$output_root/$material/${material}_base_color.png" "$thumbnail"
    sips -z 48 48 "$thumbnail" >/dev/null
    bash "$block_thumbnail_renderer" \
        "$thumbnail" \
        "$output_root/$material/${material}_block_thumbnail.png"
done

for material in "${materials[@]}"; do
    for map in "${maps[@]}"; do
        output="$output_root/$material/${material}_${map}.png"
        read -r width height < <(image_dimensions "$output")
        [[ $width == 3072 && $height == 3072 ]] || exit 1
    done
    thumbnail="$output_root/$material/${material}_thumbnail.png"
    read -r width height < <(image_dimensions "$thumbnail")
    [[ $width == 48 && $height == 48 ]] || exit 1
    block_thumbnail="$output_root/$material/${material}_block_thumbnail.png"
    read -r width height < <(image_dimensions "$block_thumbnail")
    [[ $width == 96 && $height == 106 ]] || exit 1
done

if $check_only; then
    printf 'validated %d maps and %d flat plus isometric thumbnails; normalized %d source maps; no files changed\n' \
        "$(( ${#materials[@]} * ${#maps[@]} ))" "${#materials[@]}" "$normalized_count"
    exit 0
fi

changed_count=0
for material in "${materials[@]}"; do
    mkdir -p "$asset_root/$material"
    for filename in "${material}_base_color.png" "${material}_normal.png" \
        "${material}_orm.png" "${material}_thumbnail.png" \
        "${material}_block_thumbnail.png"; do
        source="$output_root/$material/$filename"
        target="$asset_root/$material/$filename"
        if ! cmp -s "$source" "$target"; then
            install -m 0644 "$source" "$target"
            printf 'updated %s/%s\n' "$material" "$filename"
            ((changed_count += 1))
        fi
    done
done

printf 'installed %d changed assets; normalized %d source maps\n' \
    "$changed_count" "$normalized_count"
