#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'usage: %s INPUT_PNG OUTPUT_PNG\n' "${0##*/}" >&2
    exit 2
fi

input=$1
output=$2
[[ -f $input ]] || {
    printf 'input image not found: %s\n' "$input" >&2
    exit 1
}
command -v ffmpeg >/dev/null || {
    printf 'required command not found: ffmpeg\n' >&2
    exit 1
}

# Project one square material sample onto three faces of a 96 x 106 isometric
# block. The alpha masks remove the perspective filter's pixels outside each
# quadrilateral; different face exposure keeps the volume readable without
# changing the source material's hue.
ffmpeg -hide_banner -loglevel error -y -i "$input" -filter_complex \
    "[0:v]format=rgba,scale=96:106:flags=lanczos,split=3[top][left][right];\
[top]perspective=x0=48:y0=3:x1=91:y1=28:x2=5:y2=28:x3=48:y3=53:sense=destination:eval=init,\
geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(between(X,5,91)*gte(Y,3+abs(X-48)*25/43)*lte(Y,53-abs(X-48)*25/43),255,0)'[topface];\
[left]colorchannelmixer=rr=0.74:gg=0.74:bb=0.74,\
perspective=x0=5:y0=28:x1=48:y1=53:x2=5:y2=78:x3=48:y3=103:sense=destination:eval=init,\
geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(between(X,5,48)*gte(Y,28+(X-5)*25/43)*lte(Y,78+(X-5)*25/43),255,0)'[leftface];\
[right]colorchannelmixer=rr=0.54:gg=0.54:bb=0.54,\
perspective=x0=48:y0=53:x1=91:y1=28:x2=48:y2=103:x3=91:y3=78:sense=destination:eval=init,\
geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(between(X,48,91)*gte(Y,53-(X-48)*25/43)*lte(Y,103-(X-48)*25/43),255,0)'[rightface];\
color=c=black@0.0:s=96x106,format=rgba[base];\
[base][leftface]overlay=format=auto[leftblock];\
[leftblock][rightface]overlay=format=auto[sides];\
[sides][topface]overlay=format=auto" \
    -frames:v 1 "$output"
