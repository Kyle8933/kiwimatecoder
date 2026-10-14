# Media providers and models work like LLM providers and models

Date: 2026-10-14
Status: Draft, awaiting review

## Goal

`/image`, `/video` and `/media` should choose providers and models the way
`/model` and `/provider` do for LLMs: a model is remembered per provider, no
model is pre-chosen, models come from a live catalog you can list, refresh and
search, and an interactive picker is offered wherever a choice is needed.

## Scope

In scope ("selection parity"):

- Per-provider, per-kind (image, video) remembered models.
- No built-in default model.
- A live, cached model catalog for image and video models, with list, refresh
  and search.
- Interactive pickers for media models and for the media provider.
- The same behaviour in the slash commands, the `config media` CLI, and the
  `generate_image` / `generate_video` tools.

Out of scope:

- An ordered roster of active media providers with fallback. Retrying a paid
  media request on another provider risks double billing.
- Separate providers for images and for video. `media.provider` stays a single
  provider shared by both kinds.

## Current behaviour (what changes)

| | LLM | Media today |
|---|---|---|
| Model storage | `provider_models`, per provider | `media.model`, `media.video_model`, global |
| Default model | none | `gpt-image-1` for images |
| Choosing a model | `/model` picker, `list`, `refresh`, `search`, live catalog | free text: `--model`, `/config media model` |
| Choosing a provider | `/provider` picker | `media.provider` string, set via `/config media provider` |
| Catalog | `get_model_catalog`, chat models only | none; image and video ids are filtered out of the chat catalog |

## Design

### 1. Config (`config.py`)

- Models are stored in `media.models` as `{provider_id: {"image": str, "video": str}}`.
  An absent kind means no model is chosen.
- `media.model` and `media.video_model` stay readable for existing configs.
  `get_media()` treats a legacy value as the choice for `media.provider` when
  `media.models` has no entry for that provider and kind. The next
  `set_media*` call writes `media.models` and drops the legacy keys.
- `MEDIA_DEFAULTS["model"]` becomes `""`. The default `media` block in the
  config template changes to match.
- `get_media()` and `set_media()` keep their signatures and return shapes.
  `get_media()["model"]` and `["video_model"]` hold the effective model for the
  current provider, so existing callers keep working.
- New helpers mirror `get_provider_model` / `set_provider_model`:
  `get_media_model(provider_id, kind, cfg=None) -> str` and
  `set_media_model(provider_id, kind, model | None)`. Both validate the
  provider id (`UnknownProviderError`) and the kind.
- `validate_config` checks `media.models` shape (object of objects of strings)
  and keeps checking the legacy keys.
- Malformed `media.models` values are ignored by `get_media()`, never raised,
  consistent with the other media settings.

### 2. Catalog (`catalog.py`, `config.py`)

- `get_model_catalog`, `list_visible_models` and `search_model_catalog` gain
  `kind: Literal["chat", "image", "video"] = "chat"`. Chat behaviour is
  unchanged.
- The model cache is keyed `provider_id` for chat and `provider_id:image` /
  `provider_id:video` for media. Existing caches stay valid.
- TTL, failure backoff, `refresh`, `force`, `keep`, `limit` and the live versus
  cache versus curated fallback apply to every kind.
- For `image` and `video`, `parse_models_response` uses a media filter in place
  of `is_chat_model`. The filter keeps a model when:
  - the entry reports `architecture.output_modalities` containing the kind, or
  - the entry has no modality metadata and the id matches that kind's name
    markers (`image`, `imagen`, `flux`, `dall-e`, `diffusion` for images;
    `video`, `veo`, `sora` for video).
- The allow/deny model filter (`apply_model_filter`) applies per provider to
  media lists as well.
- Providers have no curated media list, so the curated fallback is empty. A
  provider with no key and no cache shows an empty list with the usual
  "choose by name" hint.
- Risk: only some providers (OpenRouter) report modalities. For others a media
  model whose id does not match the markers will not be listed. Typing an id
  directly always works, as it does for LLMs.

### 3. Commands (`commands.py`, `main.py`)

- `/image model [list|refresh|search <term>|<id>]` and
  `/video model [list|refresh|search <term>|<id>]` mirror `/model`. With a
  selector available, `list` and `search` open a picker with the current model
  pinned. A chosen id is saved with `set_media_model(provider, kind, id)`.
  With no argument, print the current model.
- `/media provider [id]` mirrors `/provider`. With no argument it shows a table
  of providers with their chosen image and video model. With an id it sets
  `media.provider`, then, for each kind with no model chosen for that provider
  and a selector available, opens that kind's model picker. Without a selector
  it prints the hint `/image model` or `/video model`.
- `/image` and `/video` with no model chosen open the model picker when a
  selector is available. Otherwise they print a no-model message that names the
  fix command, in the style of `no_model_message`. `--model <id>` stays as a
  per-call override and is not saved.
- Existing `/config media …` forms keep working. `/config media model` and
  `/config media video-model` accept an optional trailing provider id, like
  `config model set <model> --provider`, and default to `media.provider`.
- `config media model`, `config media video-model` and `config media provider`
  in the CLI get the same behaviour. `config media show` and `/media` list the
  chosen models per provider.
- Command help, tab completion and the slash-command coverage list include the
  new subcommands.

### 4. Tools and readiness (`media.py`, `tools/image.py`)

- `media.generate_image` and `media.generate_video` resolve the model as:
  explicit argument, else `get_media_model(provider, kind)`. An empty result
  raises `MediaError` with the fix command. The existing
  `NO_VIDEO_MODEL_MESSAGE` generalises to both kinds.
- `media.readiness()` reports a missing image model as a problem, as it already
  does for video.
- The `generate_image` / `generate_video` tool descriptions and error text
  point at `/image model` and `/video model`.

## Error handling

- Unknown provider or kind: `ValueError` / `UnknownProviderError`, shown in red
  as other config errors are.
- Catalog fetch failure: falls back to cache, then to an empty list, and the
  error is reported on the catalog result and shown by `refresh`, as for chat.
- Offline mode and proxy or CA settings apply because the fetch reuses
  `catalog.fetch_models` and the provider registry transport.

## Testing

- Config: legacy `media.model` / `media.video_model` migrate on read and on
  write; `media.models` round-trips; malformed values fall back; validation
  reports bad shapes.
- Catalog: media filter by modalities and by name markers; chat catalog
  unchanged; media cache keys do not collide with chat; refresh, force, backoff
  and search work per kind.
- Commands: `/image model` and `/video model` list, refresh, search, direct set
  and picker; `/media provider` table and set; `/image` and `/video` with no
  model; `--model` override is not persisted; `/config media` forms with an
  optional provider.
- Tools: no-model errors name the command; stored models are used.
- Existing `test_media.py`, `test_media_video.py`, `test_catalog.py`,
  `test_config*.py`, `test_slash_command_coverage.py` keep passing, with
  updates only where they assert the removed `gpt-image-1` default.

## Open items

None blocking. Per-kind providers and a media provider fallback roster can be
added later without changing the storage shape above.
