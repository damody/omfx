# omfx Web/WASM executor

## Prerequisites

```powershell
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
cargo install basic-http-server
```

## Build, stage, and run

From repo root:

```powershell
.\run_web.bat
```

This builds the script DLL, native backend, WebSocket bridge, and WASM executor; stages a static web root; starts `omobab`, `omb-ws-bridge`, and a static HTTP server; then opens the browser.

By default this uses a fast dev WASM build (`wasm-pack build --target web --dev --no-opt`) and does not run `wasm-opt`.

For an optimized release WASM build, use:

```powershell
.\run_web.bat --release
```

To build and stage without starting processes:

```powershell
.\run_web.bat --build-only
```

Options can be combined:

```powershell
.\run_web.bat --build-only --release
```

The staged static web root is:

```text
omfx/executor-wasm/web-root
```

The staged root contains `index.html`, `main.js`, `pkg/`, and `data/` copied from `omfx/data`.

## Manual backend and WebSocket bridge

The browser cannot connect to the native KCP/UDP server directly. Start the native `omb` backend, then start the bridge:

```powershell
cargo run --manifest-path omb-ws-bridge/Cargo.toml -- 127.0.0.1:50062 127.0.0.1:50061
```

The bridge exposes `ws://127.0.0.1:50062` and forwards raw `[tag][len][payload]` protobuf frames to the existing KCP server at `127.0.0.1:50061`.

## Manual static server

```powershell
basic-http-server omfx/executor-wasm/web-root
```

Open:

```text
http://localhost:4000/?omoba_ws=ws://127.0.0.1:50062&player=web-player
```

The current Web client is a WASM-safe diagnostic renderer. It initializes Fyrox, displays connection status, sends `SubscribeRequest` and `JoinRequest`, and reports `GameStart`, `TickBatch`, `GameEvent`, `StateHash`, and bridge/decode errors.

## Endpoint configuration

- `omoba_ws`: WebSocket endpoint, default `ws://127.0.0.1:50062`
- `player`: player name sent in `SubscribeRequest` / `JoinRequest`, default `web-player`
