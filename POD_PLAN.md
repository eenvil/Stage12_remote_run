# Stage12 8-Pod shard plan

Assuming 16 tarballs:

```text
Pod 1: SHARDS="1 9"
Pod 2: SHARDS="2 10"
Pod 3: SHARDS="3 11"
Pod 4: SHARDS="4 12"
Pod 5: SHARDS="5 13"
Pod 6: SHARDS="6 14"
Pod 7: SHARDS="7 15"
Pod 8: SHARDS="8 16"
```

Typical pod command:

```bash
git clone <your-repo-url> Stage12_remote_run
cd Stage12_remote_run

bash scripts/setup_runpod.sh

BASE_URL="https://xxxxx.trycloudflare.com" SHARDS="1 9" bash scripts/download_shards.sh

SHARDS="1 9" MESH_STRIDE=2 bash scripts/run_worker.sh

RUN_NAME="pod1" bash scripts/pack_results.sh
```

Smoke test:

```bash
BASE_URL="https://xxxxx.trycloudflare.com" SHARDS="1" bash scripts/download_shards.sh
SHARDS="1" LIMIT=1 OVERWRITE=1 DEBUG=1 bash scripts/run_worker.sh
RUN_NAME="smoke_pod1" INCLUDE_WORK=1 bash scripts/pack_results.sh
```
