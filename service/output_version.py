"""Version stamp for the pipeline result artifact's schema + semantics.

Bump :data:`PIPELINE_OUTPUT_VERSION` whenever the **shape or meaning of the
result GeoJSON changes** — columns added / removed / renamed, verdict label
format, classifier-only column pruning, etc.

Why: result artifacts are immutable (written once to MinIO, never rewritten) and
cached by input identity via task idempotency keys. Without a version in the key,
a fix to the output format leaves old cached results served as-is until someone
manually recomputes. The version is mixed into every idempotency key, so after a
bump the previously computed (now stale-format) result no longer satisfies a
cache hit and the next request recomputes automatically — no manual
recompute / backfill.

Trade-off: a bump forces a one-off recompute wave for inputs requested again
(that is the point). Keep bumps deliberate and paired with an output change.

History:
  - v1  pre-2026-07-03: raw machine verdict (``allowed_main``), untrimmed
        urban_api passthrough attributes in the scenario flow.
  - v2  2026-07-03 (f9868f7): Russian ``Вердикт_ПЗЗ`` label, trimmed result
        columns, classifier-only PZZ-column pruning.
  - v3  2026-07-09: building runner adds the ``Основание_подбора_ВРИ`` result
        column (how the ВРИ was resolved — floors / service / object type).
  - v4  2026-07-21: classifier-only runs (``include_pzz_check=False``) no
        longer expose ``Вердикт_ПЗЗ`` — PZZ zones never participate in that
        mode, so the same source column is renamed ``Статус_классификации``
        instead. The classify-only "reason" text for the classifier-candidate
        rows also no longer says "ПЗЗ отключена". PZZ-check / building runs
        are unaffected (``Вердикт_ПЗЗ`` unchanged).
  - v5  2026-09-24: scenario checks also include the scenario's services and
        fill ``Категория_объекта`` / basis column like the file building
        check, so the result splits into «здания» / «сервисы» layers.
        Building / scenario results describe objects by usage type, not ВРИ:
        verdict reasons and basis text say «тип использования», and the columns
        ``ВРИ_ЕГРН`` / ``Код_подобранного_ВРИ`` / ``Подобранный_ВРИ`` /
        ``Основание_подбора_ВРИ`` are renamed ``Исходный_тип_объекта`` /
        ``Код_типа_использования`` / ``Тип_использования`` /
        ``Основание_подбора_типа_использования`` (parcel runs unchanged).
"""

PIPELINE_OUTPUT_VERSION = "v5"
