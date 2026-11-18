#!/usr/bin/env bash
# Synthetic/empty-workspace runtime check. Does not download weights or need a GPU.
set -euo pipefail
image=${1:?Supply the CPU image tag}
workspace=$(mktemp -d)
name="ot-smoke-$$"
cleanup() { docker rm -f "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
mkdir -p "$workspace/"{videos,models,results,preferences}
docker run -d --name "$name" --init --user "$(id -u):$(id -g)" --read-only --tmpfs /tmp \
    --mount "type=bind,src=$workspace/videos,dst=/videos,readonly" \
    --mount "type=bind,src=$workspace/models,dst=/models,readonly" \
    --mount "type=bind,src=$workspace/results,dst=/results" \
    --mount "type=bind,src=$workspace/preferences,dst=/preferences" \
    -p 127.0.0.1::6080 "$image" >/dev/null
port=$(docker port "$name" 6080/tcp)
for ((i=0;i<90;i++)); do
    if [[ $(docker inspect --format '{{.State.Health.Status}}' "$name") == healthy ]]; then break; fi
    [[ $(docker inspect --format '{{.State.Running}}' "$name") == true ]] || { docker logs "$name"; exit 1; }
    sleep 1
done
[[ $(docker inspect --format '{{.State.Health.Status}}' "$name") == healthy ]]
curl --fail --silent "http://$port/api/status"
# A cross-origin stop request must be rejected.
[[ $(curl --silent -o /dev/null -w '%{http_code}' -X POST -H 'Origin: http://example.invalid' "http://$port/api/stop") == 403 ]]
curl --fail --silent -X POST -H "Origin: http://$port" -H 'X-OrganoidTracker: local' "http://$port/api/stop"
[[ $(docker wait "$name") == 0 ]]
[[ -s $workspace/preferences/desktop.ini ]]
printf '\nCPU desktop readiness, origin rejection, graceful stop and preferences passed.\n'
