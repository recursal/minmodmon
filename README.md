# Mini Model Daemon

Quickly run an OpenAI-compatible chat completions endpoint, using RWKV language models (such as EagleX).

## Getting Started

> This guide is specifically for Windows users.
> minmodmon works on other platforms too, but no pre-compiled binaries are available currently.

1. Download minmodmon from the [latest stable release](https://github.com/recursal/midmodmon/releases), or from the
   [latest build](https://github.com/recursal/midmodmon/actions).
2. Unzip the archive, and run "minmodmon-server.exe".
3. Open http://localhost:5000/ in your web browser, to open the dashboard.
4. Download the model you want to use from the download link in the dashboard.
5. Place the ".st" model file in the "data" directory.
6. Re-start "minmodmon-server.exe".
7. Under "Load Model", select the ID of the model you downloaded.
8. Press "Load". Refresh the page to check loading status, until "Loaded model" changes to the model ID.

### SillyTavern

1. In the top menu bar, press the "API" button.
2. Select "Chat Completion" API type.
3. Under "Chat Completion Source", select "Custom (OpenAI-compatible)"
4. Under "Custom Endpoint (Base URL)", enter "http://localhost:5000/api"
5. Press "Connect".
6. Under "Available Models", select your previously loaded model. If only "None" is available, follow the instructions
   above to load a model.
