# RespectASO

<p align="center">
  <img src="desktop/assets/RespectASO.iconset/icon_256x256.png" alt="RespectASO" width="128">
</p>

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![macOS](https://img.shields.io/badge/macOS-Download_.dmg-purple?logo=apple&logoColor=white)](https://github.com/respectlytics/respectaso/releases/latest)
[![Version](https://img.shields.io/github/v/release/respectlytics/respectaso?color=purple&label=version)](https://github.com/respectlytics/respectaso/releases/latest)

**Free, open-source ASO keyword research tool for macOS. No API keys. No accounts. No data leaves your machine.**

RespectASO helps iOS developers research App Store keywords privately. Download the `.dmg`, drag to Applications, and get keyword popularity scores, difficulty analysis, competitor breakdowns, and download estimates — all without sending your research data to third-party services.

---

## Why RespectASO?

Most ASO tools require paid subscriptions, API keys, and send your keyword research to their servers. RespectASO takes a different approach:

- **No API keys or credentials needed** — uses only the public iTunes Search API
- **Runs entirely on your machine** — all API calls originate from your local network
- **No telemetry, no analytics, no tracking** — zero data sent to any third party
- **Free and open-source** — AGPL-3.0 licensed, forever
- **Native Mac app** — download the `.dmg`, drag to Applications, done

## Features

| Feature | Description |
|---------|-------------|
| **Keyword Popularity** | Estimated popularity scores (1–100) based on analysis of iTunes Search API competitor data |
| **Difficulty Score** | Competition difficulty analysis across multiple factors with ranking tier breakdowns for Top 5, Top 10, and Top 20 |
| **Ranking Tiers** | Separate difficulty analysis for Top 5, Top 10, and Top 20 positions — because breaking into the top 5 is different from reaching the top 20 |
| **Download Estimates** | Estimated daily downloads per ranking position based on search volume, tap-through rates, and conversion rates |
| **Competitor Analysis** | See the top 10 apps ranking for each keyword with ratings, reviews, genre, release date, and direct App Store links |
| **Country Opportunity Finder** | Scan up to 30 App Store regions at once to find which countries offer the best ranking opportunities for your keyword |
| **Multi-Keyword Search** | Research up to 20 keywords at once (comma-separated) |
| **Multi-Country Search** | Search the same keyword across multiple countries simultaneously |
| **App Rank Tracking** | Add your apps and see where you rank for each keyword alongside competitor data |
| **Search History** | Browse past keyword research with sorting, filtering, and expandable detail views |
| **Append-Only Trend History** | Every refresh can be stored as a new snapshot (including same-day runs) for richer trend analysis |
| **Daily Auto Refresh (Cron)** | Refresh all tracked keyword-country pairs once per day via management command (recommended for Dokploy cron) |
| **AI Copilot (Apple-only)** | On-demand keyword recommendations + metadata variants using OpenAI, tracked history, and optional App Store Connect analytics |
| **CSV Export** | Export your keyword research data for use in spreadsheets |
| **ASO Targeting Advice** | Automatic keyword classification (Sweet Spot, Good Target, Hidden Gem, High Competition, Moderate, Low Volume, Avoid) based on opportunity scoring |

## Quick Start

### 1. Download

**→ [Download RespectASO.dmg](https://github.com/respectlytics/respectaso/releases/latest)** (macOS 12+, Apple Silicon)

### 2. Install

Open the `.dmg` and drag **RespectASO** into your **Applications** folder.

### 3. Launch

Open RespectASO from Applications (or Spotlight: ⌘ Space → "RespectASO"). The app window opens automatically — type a keyword, select a country, and click Search.

> **First launch:** If macOS shows a security dialog, right-click the app → Open → Open. This is only needed once — the app is code-signed and notarized by Apple.

### Updating

When an update is available, a banner appears on the Dashboard with release notes and a **Download Update** button. Download the new `.dmg`, drag to Applications (replace the old version), and relaunch. Your data is preserved — it lives in `~/Library/Application Support/RespectASO/`, separate from the app bundle.

### Data Location

Your keywords, search history, and settings are stored at:

```
~/Library/Application Support/RespectASO/
```

This data survives app updates and deletions. Delete this folder only if you want a completely fresh start.

<details>
<summary><strong>🐳 Docker (free features only)</strong></summary>

Docker provides the **free edition** of RespectASO (keyword research, difficulty scoring, ranking tracking). AI-powered Pro features require the native macOS app above.

#### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running

#### Install via Docker

```bash
git clone https://github.com/respectlytics/respectaso.git
cd respectaso
docker compose up -d
```

Open **[http://localhost:8088](http://localhost:8088)** in your browser.

That's it. The first startup takes a few seconds (database migration + static files).

On startup, the tool automatically:
- Generates a secure Django secret key
- Checks for pending database migrations and applies them if needed
- Collects static files
- Starts the Gunicorn server

You'll see the RespectASO dashboard ready to search. Type a keyword, select a country, and click Search.

## How Scoring Works

RespectASO uses the **iTunes Search API** as its only data source — no Apple Search Ads credentials, no scraping, no paid APIs.

### Popularity Score (1–100)

A 6-signal composite model that estimates how often a keyword is searched:

| Signal | Weight | What It Measures |
|--------|--------|------------------|
| Result count | 0–25 pts | How many apps appear for this keyword |
| Leader strength | 0–30 pts | Rating volume of the top-ranking apps |
| Title match density | 0–20 pts | How many apps use this exact keyword in their title |
| Market depth | 0–10 pts | Whether strong apps appear deep in results |
| Specificity penalty | -5 to -30 | Adjusts for generic terms that inflate result counts |
| Exact phrase bonus | 0–15 pts | Rewards multi-word keywords with precise matches |

### Difficulty Score (1–100)

A 7-factor weighted system that estimates how hard it is to rank:

| Factor | Weight | What It Measures |
|--------|--------|------------------|
| Rating volume | 30% | How many ratings competitors have |
| Dominant players | 20% | Whether a few apps dominate (100K+ ratings) |
| Rating quality | 10% | Average star ratings of competitors |
| Market maturity | 10% | How long competitors have been on the App Store |
| Publisher diversity | 10% | Whether results come from many publishers or a few |
| App count | 10% | Total number of relevant results |
| Content relevance | 10% | How well competitors match the keyword |

**Interpretation:** Very Easy (&lt;16) · Easy (16–35) · Moderate (36–55) · Hard (56–75) · Very Hard (76–90) · Extreme (91+)

### Download Estimates

A 3-stage pipeline estimates daily downloads per ranking position:

1. **Popularity → Daily Searches** — piecewise-linear mapping calibrated against real App Store observations
2. **Position → Tap-Through Rate** — power-law decay from position #1 (30%) to position #20 (0.06%)
3. **Tap → Install Conversion** — range of 35%–55% for free apps

Results are shown as conservative–optimistic ranges per position, with tier breakdowns for Top 5, Top 6–10, and Top 11–20.

For more details, visit the **Methodology** page inside the app.

## Configuration

### Changing the Port

The default `docker-compose.yml` maps port **8088** on your host to port 8080 in the container. If port 8088 is already in use:

```yaml
ports:
  - "9090:8080"  # Access at http://localhost:9090
```

### Custom Local Domain

For a cleaner URL, add this to your `/etc/hosts` file:

**macOS / Linux:**
```bash
sudo sh -c 'echo "127.0.0.1  respectaso.private" >> /etc/hosts'
```

**Windows (run as Administrator):**
```
echo 127.0.0.1  respectaso.private >> C:\Windows\System32\drivers\etc\hosts
```

Then access the tool at **[http://respectaso.private](http://respectaso.private)**

The `.private` TLD is reserved by [RFC 6762](https://www.rfc-editor.org/rfc/rfc6762) and avoids conflicts with macOS mDNS resolution (unlike `.local`).

### Custom Hosts / CSRF Origins (Cloudflare Tunnel)

Set these env vars (comma-separated) in your `.env` file or shell before `docker compose up`:

```bash
ALLOWED_HOSTS=localhost,127.0.0.1,respectaso.private,aso.slusk.org
CSRF_TRUSTED_ORIGINS=http://localhost,http://127.0.0.1,http://respectaso.private,https://aso.slusk.org
```

Notes:
- `ALLOWED_HOSTS` must contain hostnames only (no scheme).
- `CSRF_TRUSTED_ORIGINS` must contain full origins (with `http://` or `https://`).

### AI + Refresh Settings

Set these in `.env` for AI Copilot, App Store Connect sync defaults, and retention/scheduler behavior:

```bash
RESULT_RETENTION_DAYS=365
AUTO_REFRESH_MODE=external
SQLITE_TIMEOUT_SECONDS=20

OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-mini
OPENAI_AVAILABLE_MODELS=gpt-5.2,gpt-5-mini,gpt-5-nano,gpt-5,gpt-4.1-mini,gpt-4.1
OPENAI_TIMEOUT_SECONDS=40
OPENAI_MAX_RETRIES=2
AI_MAX_CANDIDATES=12
AI_EVALUATED_CANDIDATES=8
AI_MAX_COUNTRIES=3
AI_HISTORY_ROWS_MAX=300
AI_ENABLE_ONLINE_CONTEXT=true
AI_ONLINE_TOP_APPS_PER_COUNTRY=20
ASC_TIMEOUT_SECONDS=30
ASC_MAX_RETRIES=2
ASC_DEFAULT_DAYS_BACK=30
ASC_JWT_TTL_MINUTES=20
```

`AUTO_REFRESH_MODE=external` is recommended in production (Dokploy/cron).
`AUTO_REFRESH_MODE=thread` keeps the in-process hourly checker fallback.
You can also manage OpenAI key/model settings, prompt templates, online context controls, and App Store Connect credentials in-app at `/config/` (stored in SQLite under `/app/data`).
Prompt template placeholders: `{{SNAPSHOT_JSON}}` and `{{ONLINE_CONTEXT_JSON}}`.

### App Store Connect Integration (Optional)

AI Copilot can include your own App Store Connect analytics signals (impressions, page views, app units, conversion, proceeds) in scoring.

1. In App Store Connect, create an API key and copy:
- `Issuer ID`
- `Key ID`
- private key PEM content
2. Add your App Store Connect app ID to each app in RespectASO (`Apps` page, `ASC App ID` field). Apps added via App Store lookup automatically fall back to `track_id` if ASC App ID is blank.
3. Enter credentials on the RespectASO `Config` page.
4. Use **Sync ASC** in AI Copilot before running recommendations.

### Daily Cron Refresh (Recommended)

Run once daily (example: 03:00 server time):

```bash
python manage.py refresh_tracked_keywords --trigger cron
```

Dokploy cron example:

```cron
0 3 * * * python manage.py refresh_tracked_keywords --trigger cron
```

### Data Persistence

Your data is stored in a Docker volume (`aso_data`). Your database and secret key survive container restarts and rebuilds.

To back up your data:
```bash
docker cp respectaso-web-1:/app/data ./backup
```

#### Updating (Docker)

```bash
cd respectaso
git pull
docker compose down
docker compose build --no-cache
docker compose up -d
```

#### Migrating from Docker to Native App

Your existing data carries over. Run this one-time migration:

```bash
curl -fsSL https://raw.githubusercontent.com/respectlytics/respectaso/main/desktop/migrate-from-docker.sh | bash
```

Then install the native app and verify your data is intact. Once confirmed:

```bash
docker compose down     # Stop the container
docker compose down -v  # Also remove the volume (only after confirming native app works)
```

#### Refreshing Dependency Pins

To refresh dependency pins after changing `requirements.in`, run in a Python 3.14 environment:

```bash
pip-compile --upgrade requirements.in
```

</details>

## How Scoring Works

RespectASO uses the **iTunes Search API** as its only data source — no Apple Search Ads credentials, no scraping, no paid APIs.

### Popularity Score (1–100)

Estimates how frequently a keyword is searched by analyzing multiple signals from iTunes Search results, including the number and quality of competing apps, keyword relevance patterns, and market depth. Higher scores mean more people are searching for that keyword.

### Difficulty Score (1–100)

Estimates how hard it would be to rank for a keyword by evaluating competition strength across factors like existing app ratings, market dominance, publisher diversity, and content relevance.

**Tiers:** Very Easy (&lt;16) · Easy (16–35) · Moderate (36–55) · Hard (56–75) · Very Hard (76–90) · Extreme (91+)

### Download Estimates

Estimates daily downloads per ranking position based on search volume, expected tap-through rates by position, and install conversion rates. Results are shown as conservative–optimistic ranges with tier breakdowns for Top 5, Top 6–10, and Top 11–20.

For full methodology details, visit the **Methodology** page inside the app or explore the [source code](https://github.com/respectlytics/respectaso).

## Configuration

<details>
<summary><strong>Custom Local Domain (Docker only)</strong></summary>

If running via Docker, you can use a cleaner URL. Add this to your `/etc/hosts` file:

```bash
sudo sh -c 'echo "127.0.0.1  respectaso.private" >> /etc/hosts'
```

Then access the tool at **[http://respectaso.private](http://respectaso.private)**

The `.private` TLD is reserved by [RFC 6762](https://www.rfc-editor.org/rfc/rfc6762) and avoids conflicts with macOS mDNS resolution (unlike `.local`).

</details>

## Tech Stack

- **Python 3.14** + **Django 6.0**
- **pywebview** — native macOS WebKit window
- **SQLite** — local single-user database
- **wsgiref** — built-in Python WSGI server
- **WhiteNoise** — efficient static file serving
- **OpenAI API (optional)** — AI keyword suggestion drafts
- **Tailwind CSS** (CDN) — dark theme UI
- **PyInstaller** — macOS `.app` bundle

## Privacy

RespectASO is designed with privacy as a core principle:

- **100% local** — the tool runs entirely on your machine as a native app
- **No accounts** — no registration, no login, no user tracking
- **No telemetry** — zero analytics, zero phone-home, zero data collection
- **No API keys required for core ASO** — the core keyword pipeline uses only Apple public APIs
- **Optional OpenAI integration** — only used when you trigger AI Copilot and provide `OPENAI_API_KEY`
- **Your data stays yours** — keyword research, competitor analysis, and search history never leave your network

We built RespectASO because we believe developers should be able to research keywords without handing their competitive intelligence to a third party.

## License

[AGPL-3.0](LICENSE) — free to use, modify, and distribute. If you modify and deploy RespectASO as a service, you must share your changes under the same license.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## Contact

[respectaso@loheden.com](mailto:respectaso@loheden.com)

---

**Built by [Respectlytics](https://respectlytics.com/?utm_source=respectaso&utm_medium=readme&utm_campaign=oss)** — Privacy-focused mobile analytics for iOS & Android. We help developers avoid collecting personal data in the first place.
