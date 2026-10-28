# Changelog

Every PR into `dev` is merged with a new version and adds its section here (see
`.github/workflows/version-bump.yml`); a release to `main` tags the version that came from `dev`.

## v0.0.2 (2026-10-28)

[fix/chat-disable-vllm-thinking](https://github.com/IDUclub/PzzCompareAPI/pull/79) (#79)

- fix(chat): disable model thinking in the vLLM chat stream

## v0.0.1 (2026-10-28)

[fix/uploads-dir-writable-startup-check](https://github.com/IDUclub/PzzCompareAPI/pull/77) (#77)

- fix(uploads): fail fast when storage dirs are not writable
- fix(worker): abort startup when storage dirs are not writable
- fix(deploy): pre-create uploads dir and share it across services

