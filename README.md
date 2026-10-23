# Call Coach

A live checklist for interviews and discovery calls. Load your questions from a Markdown file, press Start, and Call Coach listens to both sides of the call, transcribes it on your Mac, and checks off each question within seconds of it being answered. It highlights the best question to ask next, and any AI assistant that speaks MCP can push follow-up questions onto the page while you talk.

## How it works

1. **Capture.** A small Swift helper uses Apple's ScreenCaptureKit to record your microphone and the call's audio as two separate streams. Separate streams mean the transcript knows who said what: the microphone is **You**, the system audio is **Them**.
2. **Transcribe.** [whisper.cpp](https://github.com/ggml-org/whisper.cpp) runs locally on your Mac. Audio is never saved and never leaves your machine.
3. **Check off.** Every few seconds, the recent transcript and your open questions go to [Jev](https://docs.typesafe.ai) by TypeSafe AI, a fast model that scores yes/no questions. A question is ticked once Jev is confident twice in a row that it was answered. Jev also scores which open question fits best next.
4. **Follow up.** Call Coach is an MCP server. Connect Claude Code, Claude Desktop, Cursor or any MCP client, and it can follow the call and suggest follow-up questions.
5. **Save.** Press Stop and two Markdown files are written next to your questions file: the transcript, and your question list with answered items ticked and follow-ups added. Your questions file is never changed.

Jev is the only part that sends anything off your Mac: the last few minutes of transcript text and your questions. Without a Jev key, Call Coach still transcribes and you tick questions by hand.

## Requirements

- macOS 15 or later on Apple silicon
- [Homebrew](https://brew.sh)
- Headphones (see Limitations)
- A Jev API key from [console.typesafe.ai](https://console.typesafe.ai/keys), optional but needed for automatic checkmarks

## Install

    git clone https://github.com/darlison/call-coach.git
    cd call-coach
    ./scripts/install.sh            # add --login to start it automatically at login

The first run downloads the Whisper model (about 500 MB). Set your Jev key in the environment that runs Call Coach:

    export TYPESAFE_API_KEY=...
    .venv/bin/call-coach

Then open <http://localhost:4800>. In Chrome you can install the page as an app (the install icon in the address bar) so it opens in its own window.

The first time you press Start, macOS asks for **Screen Recording** and **Microphone** permission for the app that launched Call Coach (your terminal, or Python when started at login). Allow both, then press Start again.

## Questions file

Any Markdown works. Every list item, numbered or bulleted, is a question; headings group them.

    # Founder interview

    ## Early days
    1. When did you start charging customers?
    2. Who was your first paying customer?

    ## Sales
    - How many cold emails went out last month?

## Connect an AI assistant for follow-ups

The MCP endpoint is `http://localhost:4800/mcp` (Streamable HTTP). For Claude Code:

    claude mcp add call-coach --transport http http://localhost:4800/mcp

Then ask it: "Follow my call in Call Coach and suggest follow-up questions." Tools:

| Tool | What it does |
| --- | --- |
| `get_session` | Questions, which are answered, follow-ups, the transcript so far, and a cursor |
| `wait_for_update` | Waits until someone says something new after the cursor, then returns just that |
| `suggest_followup` | Puts a follow-up on the page, optionally under a numbered question |
| `get_transcript` | The full transcript as Markdown, for recaps after the call |

## Configuration

Optional, in `~/.config/call-coach/config.json`:

    {
      "jev": {"url": "https://api.typesafe.ai/v1/systemone", "api_key_env": "TYPESAFE_API_KEY"},
      "model": "~/Library/Application Support/call-coach/models/ggml-small.en.bin"
    }

To keep your Jev key off your Mac, run a small proxy that holds it and point `jev.url` at the proxy, with `"broker_secret_file"` naming a file that holds the proxy's caller secret. Call Coach sends that secret in an `x-broker-secret` header.

Set `CALL_COACH_PORT` to use a port other than 4800.

## Limitations and plans

- **Headphones are required.** Without them your microphone also hears the other person, so their words appear twice. Planned: echo filtering so speakers work.
- **Mac only.** Windows and Linux need their own audio capture; the helper in `capture/` is the only part to replace.
- **No native app yet.** Planned once the tool has been used on more calls.

## Development

    .venv/bin/python -m unittest discover -s tests -t .
    .venv/bin/call-coach --dev --questions examples/questions.md

`--dev` enables `POST /api/dev-line` (`{"speaker": "Them", "text": "..."}`) to feed transcript lines without audio.

## Licence

MIT. Whisper and whisper.cpp are MIT-licensed; the model is downloaded from Hugging Face at install time, not shipped here. Jev is a commercial API used under TypeSafe's terms with your own key.
