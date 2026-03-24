# Starburst K8s Deployment — Session Notes

**Date:** 2026-03-11 (updated)
**Environment:** Minikube on macOS (Podman driver)
**Helm release:** `sep` — chart `starburst-enterprise-479.1.0`, app version `479-e.1`
**Namespace:** `default`

---

## Final Working State

| Item | Value |
|------|-------|
| Coordinator pod | `1/1 Running`, 0 restarts |
| Worker pod | `1/1 Running`, 0 restarts |
| HTTP (UI) | `http://127.0.0.1:8080` |
| HTTPS (auth/JDBC) | `https://127.0.0.1:8443` |
| Auth | File-based: `admin:Minio123` |
| Nodes | Coordinator (scheduling enabled) + 1 worker |
| Catalogs | tpch, tpcds |
| TLS cert | `*.crjv.local` ECDSA, issued by MinIO Lab ECDSA CA, expires 2028 |

---

## Deploy Command

```bash
HELM_REGISTRY_CONFIG=~/.config/helm/registry/config.json \
helm upgrade --install sep \
  oci://harbor.starburstdata.net/starburstdata/charts/starburst-enterprise \
  --version 479.1.0 \
  -f ~/.ssh/starburst/k8s/sep-test.yaml
```

If a pod is stuck terminating after an upgrade (e.g. upgrading from a pre-fix release):
```bash
kubectl delete pod -l app=starburst --grace-period=0 --force
```
With `worker.deploymentTerminationGracePeriodSeconds: 30` in values, this should no
longer be needed — new pods terminate within ~30s.

---

## Bypassing the macOS Keychain for Helm OCI

`~/.docker/config.json` has `"credsStore": "osxkeychain"`, which causes Helm OCI
operations to fail non-interactively (keychain is locked in headless sessions).

**Fix:** credentials file at `~/.config/helm/registry/config.json`:
```json
{
  "auths": {
    "harbor.starburstdata.net": {
      "auth": "cG92LWRlbGwtZXh0ZW5zaW9uMjpEaEUzbjYzZGUyZTNFOWVo"
    }
  }
}
```
The `auth` value is `base64("pov-dell-extension2:DhE3n63de2e3E9eh")`.
Set `HELM_REGISTRY_CONFIG=~/.config/helm/registry/config.json` before every
`helm` command targeting Harbor.

---

## Kubernetes Secrets and ConfigMaps Required

```bash
# Harbor image pull secret
kubectl create secret docker-registry harbor-pull-secret \
  --docker-server=harbor.starburstdata.net \
  --docker-username=pov-dell-extension2 \
  --docker-password=DhE3n63de2e3E9eh

# Starburst license (file obtained separately)
kubectl create secret generic starburst-license \
  --from-file=starburstdata.license=/path/to/starburstdata.license

# TLS keystore (PKCS12 from *.crjv.local certs)
openssl pkcs12 -export \
  -in ~/minio/certs5/bundle.crt \
  -inkey ~/minio/certs5/private.key \
  -name starburst \
  -out /tmp/starburst.p12 \
  -passout pass:changeit

kubectl create secret generic starburst-tls \
  --from-file=keystore.p12=/tmp/starburst.p12
```

---

## Helm Chart Internals (Learned the Hard Way)

### How configs reach /etc/starburst

The chart uses an init container (`starburst-enterprise-init`) that runs:
```
starburst-conf-cli generate --values-yaml /work-dir/values.yaml \
  --config-type COORDINATOR --starburst-config-dir /etc/starburst ...
```
It reads the entire `values.yaml` (stored in the `values-yaml-secret` K8s Secret)
and generates all files into an emptyDir at `/etc/starburst`. The main coordinator
container then starts with those files already present.

**Consequence:** `configMounts` and `secretMounts` are stored in `values-yaml-secret`
but the chart template does NOT add the referenced ConfigMap/Secret as pod volumes.
They are silently ignored. Use `etcFiles.other` instead.

### Adding arbitrary files to /etc/starburst

Use `coordinator.etcFiles.other` — keys become filenames, values become file content.
The init container writes these directly into `/etc/starburst/`:

```yaml
coordinator:
  etcFiles:
    other:
      myfile.properties: |
        key=value
```

### Adding volume mounts (binary files, secrets)

`coordinator.additionalVolumes` uses a **custom Starburst schema** — not standard K8s.
It automatically generates both the pod volume AND the container volumeMount:

```yaml
coordinator:
  additionalVolumes:
    - path: /mount/path/file.ext   # mountPath in container
      subPath: file.ext            # subPath within the secret/configmap
      volume:                      # standard K8s volume source
        secret:
          secretName: my-secret
```

Top-level `additionalVolumes` follows the same schema and applies to all components.

### internalTls

`internalTls: true` (top-level) does three things:
1. Changes `discovery.uri` to `https://<release-name>:8443` (e.g. `https://sep:8443`)
2. Sets `internal-communication.https.required=true`
3. Creates an additional headless ClusterIP service named after the Helm release (`sep`)
   on ports 8080/8443 — this is what workers use to find the coordinator

It does NOT generate certs — you must provide a keystore.

The `sep` headless service is only created when `internalTls: true`. Workers register
by announcing themselves to `https://sep:8443`. The chart template comment notes that
TLS certs should ideally include `DNS:sep` as a SAN; in practice the internal HTTP
client appears to skip hostname verification when the shared secret is present.

### node-scheduler.include-coordinator

The chart default sets `node-scheduler.include-coordinator=false` in config.properties.
With 0 workers, this means **no queries can execute at all** — including the metadata
queries the Starburst console uses to discover catalogs (shows as "loading" forever).

`coordinator.node-scheduler.include-coordinator: "true"` as a YAML key is silently
ignored by the chart. The only working approach is via `coordinator.additionalProperties`:

```yaml
coordinator:
  additionalProperties: |
    node-scheduler.include-coordinator=true
```

Since additionalProperties are appended to config.properties by the init container,
the `true` value overrides the default `false`.

### terminationGracePeriodSeconds — chart copy-paste bug

`coordinator.terminationGracePeriodSeconds` is **silently ignored**.

The coordinator deployment template has a copy-paste bug: both coordinator and worker
deployments read `.Values.worker.deploymentTerminationGracePeriodSeconds` (default 300s).

**Symptom:** Every helm upgrade or pod delete causes the coordinator to hang in
`Terminating` for up to 5 minutes before kubelet force-kills it. The Recreate rollout
strategy means the new coordinator pod cannot start until the old one fully terminates,
so upgrades appear stuck.

**Fix:** Set the field that both deployments actually read:

```yaml
worker:
  deploymentTerminationGracePeriodSeconds: 30
```

This reduces the grace period to 30s for both coordinator and worker. The JVM shuts
down within ~10s under normal conditions; 30s is enough headroom without a 5-minute wait.

Note: already-running pods retain their original grace period stamped at creation time.
The fix takes effect for pods created after the upgrade.

---

## What Doesn't Work: HTTP Basic Auth

When `http-server.authentication.type=PASSWORD`, Trino rejects Basic auth
(`Authorization: Basic ...`) over plain HTTP even with
`http-server.authentication.allow-insecure-over-http=true`. The property shows
`true` at startup but still returns `"Password not allowed for insecure authentication"`.

Requests with only `X-Trino-User` header are accepted over HTTP (no password
check performed). The form-based UI login also works over HTTP.

**Conclusion:** Use HTTPS (port 8443) for all JDBC/CLI/REST API access.

---

## HTTPS Configuration

Cert source: `~/minio/certs5/` — `*.crjv.local` ECDSA cert with SANs:
`DNS:localhost`, `IP:127.0.0.1`, `DNS:*.crjv.local`

The HTTPS config and keystore mount must be applied to **both coordinator and worker**.
`coordinator.additionalProperties` and `coordinator.additionalVolumes` only affect the
coordinator pod; workers need their own identical block.

```yaml
coordinator:
  additionalProperties: |
    http-server.https.enabled=true
    http-server.https.port=8443
    http-server.https.keystore.path=/etc/starburst/tls/keystore.p12
    http-server.https.keystore.key=changeit

  additionalVolumes:
    - path: /etc/starburst/tls/keystore.p12
      subPath: keystore.p12
      volume:
        secret:
          secretName: starburst-tls

worker:
  additionalProperties: |
    internal-communication.shared-secret=<same secret as coordinator>
    http-server.https.enabled=true
    http-server.https.port=8443
    http-server.https.keystore.path=/etc/starburst/tls/keystore.p12
    http-server.https.keystore.key=changeit

  additionalVolumes:
    - path: /etc/starburst/tls/keystore.p12
      subPath: keystore.p12
      volume:
        secret:
          secretName: starburst-tls
```

Both ports are exposed via the LoadBalancer service:
```yaml
expose:
  type: "loadBalancer"
  loadBalancer:
    ports:
      http:
        port: 8080
      https:
        port: 8443
```

---

## File Auth Configuration

Password hash generated with: `htpasswd -nbB -C 10 admin Minio123`

Stored directly in `values.yaml` via `etcFiles.other` (no separate K8s secret needed):

```yaml
coordinator:
  etcFiles:
    other:
      file-authenticator.properties: |
        password-authenticator.name=file
        file.password-file=/etc/starburst/password.db
      password.db: "admin:$2y$10$ljSenrDG2lOcvXtO3mEk0eju6eXky9W9rQZukRSxnlWvVOjiWirhG\n"
```

Property key is `file.password-file` (not `file`).

When `http-server.authentication.type=PASSWORD` is set, an
`internal-communication.shared-secret` is also required:
```
internal-communication.shared-secret=<random base64 string>
```
Generate with: `openssl rand -base64 32`

---

## Worker Configuration

Workers share the node with the coordinator in this minikube setup, so resource
requests must be reduced to fit. Default worker resources match the coordinator
(2 CPU / 8Gi) which will leave the worker in `Pending` on a constrained node.

```yaml
worker:
  replicas: 1
  resources:
    requests:
      cpu: "1"
      memory: "4Gi"
    limits:
      cpu: "2"
      memory: "5Gi"
  jvm:
    maxHeapSize: "4G"
```

With `internalTls: true`, workers announce themselves to `https://sep:8443`.
The worker pod does NOT need `http-server.authentication.type=PASSWORD` or
`password-authenticator.*` — those are coordinator-only. It only needs the
shared secret and HTTPS keystore.

Verify worker registration:
```bash
# Run as a statement query and poll to completion
curl -sk -u admin:Minio123 -X POST \
  -H "Content-Type: text/plain" \
  --data "SELECT node_id, http_uri FROM system.runtime.nodes ORDER BY 1" \
  https://127.0.0.1:8443/v1/statement
```
Expected: 2 rows — one coordinator (`https://10-x-x-x.ip:8443`), one worker.

---

## Auth and Query Verification

```bash
# Public endpoint — no auth needed (200)
curl -sk https://127.0.0.1:8443/v1/info

# No credentials — 401
curl -sk -o /dev/null -w "%{http_code}\n" https://127.0.0.1:8443/v1/query

# Valid credentials — 200
curl -sk -o /dev/null -w "%{http_code}\n" -u admin:Minio123 https://127.0.0.1:8443/v1/query

# Verify TLS cert SANs
echo | openssl s_client -connect 127.0.0.1:8443 2>/dev/null | \
  openssl x509 -noout -subject -issuer -ext subjectAltName

# Trust the CA for curl (no -k needed)
curl --cacert ~/minio/certs5/ca.crt -u admin:Minio123 \
  https://localhost:8443/v1/info

# Spot-check queries (tpch.tiny = small, tpcds.sf1 = larger distributed)
# Use the Python polling loop in commands.txt for full result retrieval
# tpch.tiny.customer → expect 1500
# tpcds.sf1.store_sales → expect 2880404
```

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `sep-test.yaml` | Active Helm values — minikube SEP deployment |
| `commands.txt` | Helm upgrade command reference |
| `session-notes.md` | This file |
| `traefik-values.yaml` | Traefik Helm values — RKE2 DaemonSet deployment |
| `minio-gateway.yaml` | Gateway API resources — GatewayClass, Gateway, HTTPRoutes, BackendTLSPolicy (MinIO) |
| `sep-rke2.yaml` | Starburst Enterprise Helm values — RKE2 deployment |
| `starburst-gateway.yaml` | Gateway API resources — Gateway, HTTPRoutes, BackendTLSPolicy (Starburst) |
| `minio-aistor-install.md` | MinIO AIStor installation and configuration reference (current deployment) |
| `minio-aistor-reinstall.md` | Uninstall, verify, and fresh install tutorial (latest chart versions) |
| `sep-test.yaml.2` | Earlier attempt (service: LoadBalancer, no auth) |
| `sep-test.yaml.3` | Earlier attempt (starburstPlatformLicense, no auth) |

## Known Row Counts (Validation Reference)

| Query | Expected |
|-------|----------|
| `SELECT count(*) FROM tpch.tiny.customer` | 1,500 |
| `SELECT count(*) FROM tpcds.sf1.store_sales` | 2,880,404 |

---

---

# Phase 2 — RKE2 Production Cluster

**Date:** 2026-03-16
**Environment:** RKE2 v1.32.8 on Rocky Linux 9.5 (bare metal / VMs)
**Nodes:**
| Node | Role | IP |
|------|------|----|
| k8s | control-plane, etcd, master | 192.168.5.115 |
| k8s1 | worker | 192.168.5.94 |
| k8s2 | worker | 192.168.5.110 |
| k8s3 | worker | 192.168.5.111 |
| k8s4 | worker | 192.168.5.112 |

---

## Cluster State at Start of Phase 2

### Already Running (MinIO AIStor Stack)

| Component | Namespace | Details |
|-----------|-----------|---------|
| AIStor Object Store Operator | `aistor` | Manages MinIO instances |
| DirectPV (CSI block storage) | `directpv` | Node agent on all 5 nodes; provides PVs from raw drives |
| KES Operator | `keymanager` | Key encryption service operator |
| KES StatefulSet (3 replicas) | `my-keymanager` | Running, service `my-keymanager-keymanager:7373` |
| MinIO pool (4 pods) | `primary-object-store` | `myminio-pool-0-{0..3}`, one pod per worker |
| ca-installer DaemonSet | `kube-system` | Distributes MinIO Lab ECDSA CA to all worker nodes |

MinIO services (NodePort — pre-ingress):
- S3 API (HTTPS): `<node-ip>:31001` → pod port 443
- Console (HTTPS): `<node-ip>:31000` → pod port 9443

MinIO TLS cert: `minio-wildcard-tls-secret` — CN=minio.crjv.local, SAN=`*.crjv.local`
Issued by: MinIO Lab ECDSA CA (same CA as all other cluster certs)

### rke2-ingress-nginx Status at Phase 2 Start

`rke2-ingress-nginx` was running as a DaemonSet on all 5 nodes, holding **hostPort 80
and 443** on every node. No Ingress resources were deployed — nothing was actually using
it. It was removed as part of this phase.

---

## Ingress NGINX Retirement

The community `kubernetes/ingress-nginx` controller reached end-of-life March 31, 2026.
No further releases, CVE fixes, or security patches after this date. Running it at the
cluster edge after EOL is a high security risk.

**This affects `rke2-ingress-nginx`** — it packages the community controller.
It does NOT affect the F5/NGINX Inc. version (`nginxinc/kubernetes-ingress`).

**Decision:** Replace with Traefik + Kubernetes Gateway API.

---

## Ingress Architecture Decision

| Concern | Decision |
|---------|----------|
| Ingress controller | Traefik v3 (via Helm chart traefik/traefik v39.x) |
| Routing API | Kubernetes Gateway API — NOT Traefik-native `IngressRoute` CRDs |
| Gateway API channel | Experimental (required for `BackendTLSPolicy` v1alpha3) |
| Deployment mode | DaemonSet on workers (k8s1–k8s4), `hostPort` on 80/443 |
| Load balancer | None — clients connect directly to a worker node IP |
| TLS strategy | Termination at Traefik (wildcard cert) + re-encrypt to MinIO backend via `BackendTLSPolicy` |

**Why Gateway API over IngressRoute CRDs:**
Gateway API manifests are controller-agnostic. When this cluster eventually moves to a
different conformant controller, the route manifests require no changes.

**Why hostPort over LoadBalancer:**
No hardware/software load balancer in this environment. hostPort binds Traefik directly
to port 80/443 on every worker node. Any worker IP is a valid entry point.

---

## How rke2-ingress-nginx Was Removed

RKE2 manages built-in charts via `HelmChart` CRDs (not plain Helm releases). A direct
`helm uninstall` would be overridden by the RKE2 Helm controller on the next reconcile.

**Correct removal process:**

Step 1 — Delete the HelmChart CRD resource (triggers immediate Helm uninstall):
```bash
kubectl delete helmchart rke2-ingress-nginx -n kube-system
```

Step 2 — Prevent RKE2 from reinstalling on server restart (SSH to k8s):
```bash
echo -e "\ndisable:\n  - rke2-ingress-nginx" >> /etc/rancher/rke2/config.yaml
```

No rke2-server restart required for Step 1. Step 2 takes effect on next server restart.

---

## Traefik Installation

### Gateway API CRDs (v1.5.1 — released 2026-03-14)

`TLSRoute` remains in the experimental channel as of v1.5.1. Must use
`experimental-install.yaml`, not `standard-install.yaml`.

```bash
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.5.1/experimental-install.yaml
```

### Traefik Helm Install

```bash
helm repo add traefik https://traefik.github.io/charts
helm repo update
helm install traefik traefik/traefik \
  --namespace traefik \
  --create-namespace \
  -f traefik-values.yaml
```

### traefik-values.yaml — Key Design Points

- `deployment.kind: DaemonSet` — one Traefik pod per worker node
- `nodeSelector: node-role.kubernetes.io/worker: "true"` — workers only, not control plane
- `service.type: ClusterIP` — no LoadBalancer service (avoids perpetual "pending" state)
- `ports.web.port: 8000` + `hostPort: 80` — HTTP on all worker nodes
- `ports.websecure.port: 8443` + `hostPort: 443` — HTTPS on all worker nodes
- `providers.kubernetesGateway.enabled: true` + `experimentalChannel: true` — Gateway API
- `providers.kubernetesCRD.enabled: false` — IngressRoute CRDs disabled (not used)
- `providers.kubernetesIngress.enabled: false` — classic Ingress disabled (not used)
- `gateway.enabled: false` — chart's built-in GatewayClass/Gateway disabled; we manage our own

**Port matching note:** Traefik's Gateway API provider matches Gateway listener ports to
entrypoints by the entrypoint's **internal (container) port**, not the hostPort or
exposedPort. Gateway listeners use port 8000/8443 to match `web`/`websecure` entrypoints.
Clients connect to node:80/443 via the hostPort mapping — this is transparent to Traefik.

**Note on `ports.websecure.tls`:** Traefik chart v39.x removed `tls.enabled` from the
`ports.websecure` schema. Do not include a `tls:` sub-key under `ports.websecure` —
schema validation will reject the chart install with "additional properties not allowed".

---

## MinIO Gateway API Resources (minio-gateway.yaml)

### TLS Architecture

```
Client → Traefik (presents *.crjv.local wildcard cert, TLS terminated)
       → MinIO backend (Traefik re-encrypts using MinIO's internal cert)
```

- Traefik holds `minio-wildcard-tls-secret` and presents it to external clients
- MinIO's internal auto-cert (signed by `rke2-server-ca`) is untouched
- `BackendTLSPolicy` tells Traefik how to validate the backend cert: trust `kube-root-ca.crt`,
  expect hostname `minio.primary-object-store.svc.cluster.local`

**Why not push the wildcard cert into MinIO directly:**
The AIStor `minio-sidecar` container verifies TLS internally using cluster-internal
hostnames (e.g., `myminio-pool-0-0.myminio-hl.primary-object-store.svc.cluster.local`).
These are not covered by `*.crjv.local`. Replacing MinIO's cert with the wildcard cert
causes the sidecar's readiness probe to return 502, eventually draining all pod endpoints
and taking MinIO offline. TLS termination at Traefik avoids this entirely.

### GatewayClass

```yaml
controllerName: traefik.io/gateway-controller
```

### Gateway — `minio-gateway` in namespace `primary-object-store`

Two listeners:
- `http` on port 8000 (`protocol: HTTP`) — for HTTP→HTTPS redirect only
- `https` on port 8443 (`protocol: HTTPS`, `tls.mode: Terminate`) — TLS termination
  - `certificateRefs: minio-wildcard-tls-secret` — wildcard cert served to clients

### HTTPRoutes

| Route | Hostname | Backend Service | Backend Port |
|-------|----------|----------------|--------------|
| `minio-s3` | `rke.crjv.local` | `minio` | 443 |
| `minio-console` | `console.crjv.local` | `myminio-console` | 9443 |
| `minio-http-redirect` | both hostnames | (redirect filter) | — |

**Hostname note:** `minio.crjv.local` is reserved for a separate MinIO cluster on this
network. The RKE2-hosted MinIO instance uses `rke.crjv.local` for S3 API access.

`minio-http-redirect` — catches HTTP on port 8000, returns 301 to `https://`.

### BackendTLSPolicy

Two `BackendTLSPolicy` (v1alpha3) resources, one per backend service:

| Policy | Target Service | CA ConfigMap | Validation Hostname |
|--------|---------------|--------------|---------------------|
| `minio-s3-backend-tls` | `minio` | `kube-root-ca.crt` | `minio.primary-object-store.svc.cluster.local` |
| `minio-console-backend-tls` | `myminio-console` | `kube-root-ca.crt` | `myminio-console.primary-object-store.svc.cluster.local` |

`kube-root-ca.crt` is the auto-created ConfigMap (present in every namespace) containing
the RKE2 cluster CA cert — the same CA that signed MinIO's internal auto-generated cert.

Note: `BackendTLSPolicy` v1alpha3 is deprecated as of Gateway API v1.5.1 (use v1 when
it becomes available in your Traefik version).

### Client /etc/hosts entries

```
192.168.5.94   rke.crjv.local
192.168.5.94   console.crjv.local
```
Any worker IP (k8s1–k8s4) works — all run Traefik on hostPort 443. The entries above
point to k8s1 (192.168.5.94). This is a single point of failure for client DNS: if k8s1
goes down, update the `/etc/hosts` entry to any other worker IP. For true HA, use
round-robin DNS (multiple A records) or a floating VIP (kube-vip / keepalived).

### Verified Working (2026-03-16)

| Test | Result |
|------|--------|
| `http://rke.crjv.local` | 301 → `https://rke.crjv.local/` |
| `https://rke.crjv.local/minio/health/live` | 200 |
| `https://console.crjv.local` | 200 |
| TLS cert (external) | `*.crjv.local`, SAN `DNS:*.crjv.local`, issuer `MinIO Lab ECDSA CA` |
| MinIO pods | `2/2 Running` — sidecar healthy, internal cert untouched |

**MinIO access:**
- Console: `https://console.crjv.local` — `admin` / `password`
- S3 API: `https://rke.crjv.local`
- mc: `mc alias set rke https://rke.crjv.local admin password`

### Known Issues / Port Conflict During Install

The Traefik Helm chart v39.x defaults the built-in `traefik` dashboard entrypoint to
port **8080**. If `ports.web.port` is also set to 8080, both entrypoints collide and
all pods crash-loop with `address already in use`.

**Fix:** Use port 8000 for `web` (the chart default):
```yaml
ports:
  web:
    port: 8000   # NOT 8080 — that's reserved for the traefik dashboard entrypoint
    hostPort: 80
```

Gateway API CRD installation note: the `httproutes` CRD exceeds the 262KB annotation
limit for `kubectl apply`. Always use `--server-side` flag:
```bash
kubectl apply --server-side --force-conflicts -f experimental-install.yaml
```

---

## Starburst Enterprise on RKE2

**Deployed 2026-03-16. Verified working.**

### Resources

| File | Purpose |
|------|---------|
| `sep-rke2.yaml` | Helm values — RKE2 deployment |
| `starburst-gateway.yaml` | Gateway API resources — Gateway, HTTPRoutes, BackendTLSPolicy |

### Namespace and Secrets

```bash
kubectl create namespace starburst

# Harbor image pull
kubectl create secret docker-registry harbor-pull-secret \
  --docker-server=harbor.starburstdata.net \
  --docker-username=pov-dell-extension2 \
  --docker-password=DhE3n63de2e3E9eh \
  -n starburst

# License
kubectl create secret generic starburst-license \
  --from-file=starburstdata.license=./starburstdata.license \
  -n starburst

# TLS keystore from *.crjv.local certs
openssl pkcs12 -export \
  -in ~/minio/certs5/bundle.crt \
  -inkey ~/minio/certs5/private.key \
  -name starburst \
  -out /tmp/starburst.p12 \
  -passout pass:changeit
kubectl create secret generic starburst-tls \
  --from-file=keystore.p12=/tmp/starburst.p12 \
  -n starburst

# CA ConfigMap for BackendTLSPolicy (MinIO Lab ECDSA CA — issuer of *.crjv.local)
kubectl create configmap minio-lab-ca \
  --from-file=ca.crt=~/minio/certs5/minio_edcsa_ca.crt \
  -n starburst

# Copy wildcard TLS secret into starburst namespace (needed by the Gateway)
kubectl get secret minio-wildcard-tls-secret -n primary-object-store -o json \
  | jq 'del(.metadata.namespace,.metadata.resourceVersion,.metadata.uid,.metadata.creationTimestamp,.metadata.annotations,.metadata.managedFields)' \
  | kubectl apply -n starburst -f -
```

### Deploy

```bash
HELM_REGISTRY_CONFIG=~/.config/helm/registry/config.json \
helm upgrade --install sep \
  oci://harbor.starburstdata.net/starburstdata/charts/starburst-enterprise \
  --version 479.1.0 \
  -f sep-rke2.yaml \
  -n starburst

kubectl apply -f starburst-gateway.yaml
```

`HELM_REGISTRY_CONFIG` is still required — helm runs from macOS where the osxkeychain
issue applies regardless of cluster type.

### Access

| Endpoint | URL |
|----------|-----|
| UI / API | `https://starburst.crjv.local` |
| Credentials | `admin` / `Minio123` |

```
192.168.5.94   starburst.crjv.local   # k8s1; any worker IP works
```

### Verified Working (2026-03-16)

| Test | Result |
|------|--------|
| `https://starburst.crjv.local/v1/info` | 200 — `ACTIVE`, version `479-e.1` |
| `/v1/query` without credentials | 401 |
| `/v1/query` with `admin:Minio123` | 200 |
| TLS cert | `*.crjv.local`, issuer `MinIO Lab ECDSA CA` |
| Pods | `1/1 Running` — coordinator + 2 workers |

### Gotchas Discovered During Install

**CPU requests must be ≤ 1 on 2-CPU worker nodes.**
Nodes have 2 CPUs total; existing workloads (MinIO, KES, Traefik) consume ~0.5–0.8 CPU.
A 2 CPU request will never schedule. Use `requests: cpu: "1"`, `limits: cpu: "2"`.

**The external service is `coordinator:8443`, not `sep-starburst-enterprise`.**
With `expose.type: clusterIp` and `internalTls: true`, the chart creates:
- `starburst` ClusterIP — HTTP port 8080 only (not useful for HTTPS routing)
- `coordinator` headless — ports 8080/8443/8081
- `sep` headless — ports 8080/8443 (internal worker discovery)
- `worker` headless — ports 8080/8443/8081

Route to `coordinator:8443` for HTTPS backend. The `BackendTLSPolicy` targets the
`coordinator` service.

**`http-server.process-forwarded=true` is required when behind any reverse proxy.**
Traefik (and any proxy doing TLS termination) injects `X-Forwarded-Proto` headers.
Starburst's Jetty server rejects these by default with HTTP 406:
`"Server configuration does not allow processing of the x-forwarded-proto header"`.
Fix: add `http-server.process-forwarded=true` to `coordinator.additionalProperties`.
This only needs to be set on the coordinator (workers don't receive external traffic).

**JVM trust store required for Starburst → MinIO TLS.**
MinIO's internal certificate (both port 443 and 9000) is signed by the RKE2 cluster CA
(`kube-root-ca.crt`), not the MinIO Lab ECDSA CA. The Starburst JVM must trust this CA
to connect to MinIO's Iceberg REST catalog endpoint.

Build the trust store and create the K8s secret:
```bash
# Export the RKE2 cluster CA from the cluster
kubectl get configmap kube-root-ca.crt -n kube-system \
  --kubeconfig ~/.kube/rke2.yaml \
  -o jsonpath='{.data.ca\.crt}' > /tmp/cluster-ca.crt

# Import into a JKS trust store (keytool from any JDK)
keytool -importcert -noprompt \
  -alias rke2-cluster-ca \
  -file /tmp/cluster-ca.crt \
  -keystore /tmp/cluster-truststore.jks \
  -storepass changeit

kubectl create secret generic starburst-truststore \
  --from-file=truststore.jks=/tmp/cluster-truststore.jks \
  -n starburst
```

Mount and reference in `sep-rke2.yaml` (both coordinator and worker):
```yaml
  additionalVolumes:
    - path: /etc/starburst/tls/truststore.jks
      subPath: truststore.jks
      volume:
        secret:
          secretName: starburst-truststore
  etcFiles:
    other:
      jvm.config: |
        [chart-generated content]
        -Djavax.net.ssl.trustStore=/etc/starburst/tls/truststore.jks
        -Djavax.net.ssl.trustStorePassword=changeit
```

`jvm.config` must be provided in full (copy from a running pod first — the chart init
container generates it at startup). It cannot be appended-to; the `etcFiles.other` key
replaces the generated file entirely.

---

## MinIO AIStor Upgrade (2026-03-17)

**Prior version:** RELEASE.2025-08-13T17-08-54Z (does NOT support AIStor Tables)
**New version:** RELEASE.2026-03-12T15-15-27Z
**License:** ENTERPRISE-PLUS, valid until 2027-09-29

AIStor Tables (Iceberg REST catalog) became GA in RELEASE.2026-02-02T23-40-11Z.
Any version before that date lacks the `/_iceberg` endpoint entirely.

### Upgrade Steps

1. Find the latest objectstore chart version and image tag:
```bash
helm search repo minio/aistor-objectstore --versions | head -5
# Latest: 1.0.13
```

2. Upgrade (must supply both repository AND tag — chart validates both are present):
```bash
helm upgrade aistor-objectstore minio/aistor-objectstore \
  --version 1.0.13 \
  -n primary-object-store \
  --reuse-values \
  --set objectStore.image.repository=quay.io/minio/aistor/minio \
  --set objectStore.image.tag=RELEASE.2026-03-12T15-15-27Z
```

3. If pods do not roll (operator logs show "no such host: object-store-operator…"):
```bash
# The object-store-operator Service (port 4221) is missing — recreate it
helm upgrade aistor minio/aistor-objectstore-operator \
  --version 5.1.0 \
  --reuse-values \
  -n aistor
# Then trigger rolling restart manually
kubectl rollout restart statefulset myminio-pool-0 -n primary-object-store
```

### Verify
```bash
kubectl exec -n primary-object-store myminio-pool-0-0 -- \
  mc admin info local --insecure 2>/dev/null | grep Version
# Expected: Version: RELEASE.2026-03-12T15-15-27Z
```

---

## AIStor Tables / Iceberg REST Catalog Setup (2026-03-17)

### Create the Warehouse

**CRITICAL: `mc mb` creates a regular S3 bucket — this is NOT sufficient.**
AIStor Tables requires the warehouse to be registered via `mc table warehouse create`.
The Iceberg REST catalog's `/v1/config?warehouse=<name>` endpoint returns
`"The specified warehouse does not exist"` until this step is done.

If the bucket was already created with `mc mb` (regular bucket), use `--upgrade-existing`:
```bash
kubectl exec -n primary-object-store --kubeconfig ~/.kube/rke2.yaml myminio-pool-0-0 -- \
  mc table warehouse create local warehouse --upgrade-existing --insecure
# Output: Warehouse `warehouse` created successfully.
```

Verify:
```bash
kubectl exec -n primary-object-store --kubeconfig ~/.kube/rke2.yaml myminio-pool-0-0 -- \
  mc table warehouse ls local --insecure
# Output: warehouse
```

### Starburst aistor Catalog (sep-rke2.yaml)

```yaml
catalogs:
  aistor: |-
    connector.name=iceberg
    iceberg.catalog.type=rest
    iceberg.rest-catalog.uri=https://minio.primary-object-store.svc.cluster.local:443/_iceberg
    iceberg.rest-catalog.warehouse=warehouse
    iceberg.rest-catalog.security=SIGV4
    iceberg.rest-catalog.signing-name=s3tables
    iceberg.rest-catalog.vended-credentials-enabled=true
    iceberg.rest-catalog.view-endpoints-enabled=false
    iceberg.unique-table-location=true
    iceberg.security=system
    fs.native-s3.enabled=true
    fs.hadoop.enabled=false
    s3.endpoint=https://minio.primary-object-store.svc.cluster.local:443
    s3.path-style-access=true
    s3.region=us-east-1
    s3.aws-access-key=admin
    s3.aws-secret-key=password
```

Internal endpoint used (`minio.primary-object-store.svc.cluster.local:443`) — both
the Iceberg REST catalog and S3 data access go through the same HTTPS port.

### Verified Working (2026-03-17)

| Test | Result |
|------|--------|
| `SHOW SCHEMAS FROM aistor` | `information_schema`, `schema_discovery`, `system` |
| `CREATE SCHEMA aistor.demo` | OK |
| `CREATE TABLE aistor.demo.events (...)` | OK (Parquet format) |
| `INSERT INTO aistor.demo.events` | OK |
| `SELECT * FROM aistor.demo.events` | Returns inserted rows |

### iceberg.max-partitions-per-writer

Default is 100. Session property `max_partitions_per_writer` is hard-capped at 100 in
SEP 479 and cannot be raised via `SET SESSION`. To exceed 100 partitions in a single
CTAS/INSERT, set the catalog-level property in `sep-rke2.yaml`:

```yaml
catalogs:
  aistor: |-
    ...
    iceberg.max-partitions-per-writer=1000
```

This requires a Helm upgrade to take effect. After upgrade the catalog default is 1000
and the session property cap is no longer a bottleneck.

---

## Starburst Backend Database (Insights Persistence) (2026-03-17)

**Database:** PostgreSQL on `hms.crjv.local:5432`, database `starburst_rke`
**Credentials:** `postgres` / `postgres`

Enables persistence for:
- Query history (Insights)
- Worksheets
- Resource groups
- Session properties manager
- Login audit events

### Configuration

**There is NO top-level `database:` key in the SEP 479 Helm chart.**
Adding `database: host: ...` at the top level is silently ignored — it stores in the
`values-yaml-secret` but no chart template reads it.

The correct approach is `coordinator.additionalProperties`:

```yaml
coordinator:
  additionalProperties: |
    ...
    insights.persistence-enabled=true
    insights.jdbc.url=jdbc:postgresql://hms.crjv.local:5432/starburst_rke
    insights.jdbc.user=postgres
    insights.jdbc.password=postgres
```

### What Happens on First Start

Flyway runs 47+ schema migrations against the database automatically:
- Workload management tables (`workload_management_migrations`)
- Query history, worksheets, resource groups, session properties, login events, etc.
- Migrations are idempotent — safe to restart or re-upgrade

Confirm in coordinator logs:
```
insights.persistence-enabled    false → true
insights.jdbc.url               jdbc:postgresql://hms.crjv.local:5432/starburst_rke
HikariPool - Added connection   org.postgresql.jdbc.PgConnection@...
Flyway - Successfully applied 47 migrations to schema "public"
```

### Verify Config is Active
```bash
kubectl logs -n starburst \
  $(kubectl get pod -n starburst -l role=coordinator -o jsonpath='{.items[0].metadata.name}') \
  --kubeconfig ~/.kube/rke2.yaml \
  | grep -E "insights.persistence-enabled|insights.jdbc.url|HikariPool|Flyway"
```

---

## MinIO AIStor Chart Version Reference (2026-03-17)

### Currently Installed (This Cluster)

| Chart | Release Name | Namespace | Version |
|-------|-------------|-----------|---------|
| `aistor-volumemanager` | `directpv` | `default` / `directpv` | 0.2.0 |
| `aistor-objectstore-operator` | `aistor` | `aistor` | 5.1.0 |
| `aistor-keymanager-operator` | `keymanager-operator` | `keymanager` | 1.1.4 |
| `aistor-keymanager` | `my-keymanager` | `my-keymanager` | 2.1.0 |
| `aistor-objectstore` | `primary-object-store` | `primary-object-store` | 1.0.13 |
| MinIO image | — | — | RELEASE.2026-03-12T15-15-27Z |

### Latest Available (as of 2026-03-17)

| Chart | Latest Version | Notes |
|-------|---------------|-------|
| `minio/aistor-volumemanager` | **0.3.2** | Was 0.2.0 — minor update |
| `minio/aistor-operator` | **5.4.0** | **Replaces** `aistor-objectstore-operator`; adds webhook component |
| `minio/minkms-operator` | **1.3.0** | **Replaces** `aistor-keymanager-operator`; manages `MinKMS` CR |
| `minio/minkms` | **2.2.0** | **Replaces** `aistor-keymanager`; rebranded KMS instance chart |
| `minio/aistor-objectstore` | **1.0.13** | Unchanged — current |

### Key Breaking Changes in the New Stack

**`aistor-objectstore-operator` → `aistor-operator`**
- Single chart now manages object-store, adminjob, and new webhook operators
- Webhook (`object-store-webhook` pod) handles hot binary updates in 5.4.0
- `operators.object-store.webhook.enabled: true` required for the webhook

**`aistor-keymanager-operator` + `aistor-keymanager` → `minkms-operator` + `minkms`**
- CRD kind changes: `KeyManager` (aistor.min.io/v1alpha1) → `MinKMS`
- Helm key prefix changes: `keyManager.*` → `minkms.*`
- Service name pattern changes: `my-keymanager-keymanager` → `my-minkms-minkms`
- Namespace recommendation: `keymanager`/`my-keymanager` → `minkms`/`my-minkms`
- HSM key generation command: `docker run quay.io/minio/aistor/minkms:latest --soft-hsm`
- API key can now be auto-generated by the operator (omit `minkms.apikey.key`)

### Starburst Impact of MinKMS Rename

If reinstalling with the new `minkms` chart the MinKMS service endpoint changes:

| | Old | New |
|-|-----|-----|
| Namespace | `my-keymanager` | `my-minkms` |
| Service | `my-keymanager-keymanager.my-keymanager.svc.cluster.local:7373` | `my-minkms-minkms.my-minkms.svc.cluster.local:7373` |
| ObjectStore env | `MINIO_KMS_SERVER=https://my-keymanager-keymanager...` | `MINIO_KMS_SERVER=https://my-minkms-minkms...` |

The Starburst `sep-rke2.yaml` and `starburst-truststore` secret are unaffected —
Starburst connects to MinIO's S3 endpoint, not directly to MinKMS.

### Reference Documents

- **`minio-aistor-install.md`** — Reference guide for the current deployment. Covers
  architecture, all install steps, secrets inventory, upgrade procedure, and
  troubleshooting based on what is actually running on this cluster.

- **`minio-aistor-reinstall.md`** — Step-by-step tutorial covering complete uninstall,
  clean-state verification, and fresh install using the latest chart versions
  (`aistor-operator` 5.4.0, `minkms-operator` 1.3.0, `minkms` 2.2.0,
  `aistor-volumemanager` 0.3.2).
