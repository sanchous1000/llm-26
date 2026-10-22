# `source/` — справочник по коду

Отчёт о работе — в [../README.md](../README.md). Здесь — карта модулей и команд.
Все команды запускаются из каталога `lab2/`: `python -m source <команда> [--help]`.

## Модули

| пакет | назначение |
|---|---|
| `bibliography/` | разбор `sources.bib` и `SOURCES.md` → `Source` (ключ, метаданные, приоритет ★, теги) |
| `net/`, `downloaders/` | HTTP-клиент (паузы по хостам, повторы, атомарная запись) и загрузчики arXiv / DOI / GitHub / web |
| `storage/`, `pipeline.py`, `report.py` | состояние загрузки (`meta.json`, `manifest.json`), возобновляемый цикл, сводка |
| `parsing/`, `conversion_pipeline.py` | конвертация в Markdown: `latexml/` (arXiv HTML), `pdf`, `readme`, `webpage`; YAML front matter |
| `evaluation/` | генерация вопросов через `opencode`, проверка цитат в тексте (`locate`), отбор и экспорт |
| `indexing/` | конфиг, чанкинг (`blocks` → `units` → `splitters/`), эмбеддинги BGE-M3, PostgreSQL, Qdrant, оценка поиска, бенчмарк HNSW |
| `rag/` | ретривер, промпт и цитаты, движок, DeepSeek через `opencode`, CLI (`ask`, `chat`), MCP-сервер, `rag-eval` |
| `tests/` | pytest (188 тестов) |

## Команды

| этап | команды |
|---|---|
| 1. сбор | `download` (`--kinds`, `--keys`, `--pdf`, `--force`), `report` |
| 2. разбор | `convert` (`--force`, `--no-references`, `--workers`) |
| 3. вопросы | `gen-questions`, `review-questions`, `export-questions` |
| 4. индекс | `db-init`, `chunk`, `embed`, `index` (`--rebuild`), `index-status`, `index-drop`, `chunk-stats` |
| 5. оценка | `index-check` (`--vary`, `--lang`, `--markdown`), `hnsw-bench`, `index-apply` |
| 6. RAG | `ask`, `chat`, `search`, `rag-eval`, `mcp`, `mcp-config` |

Настройки индексации и RAG — `configs/*.yaml`, переопределение: `--set chunking.chunk_size=512`.
Секреты и порты БД — `.env` (шаблон `.env.example`), инфраструктура — `docker-compose.yml`.

## Данные (`lab2/data/`, `lab2/eval/`)

```
data/sources.bib, data/SOURCES.md       исходный список источников и каталог (приоритет ★, теги)
data/papers/                            корпус (в git не хранится)
  manifest.json                         индекс всех записей
  <kind>/<key>/meta.json                метаданные + результат загрузки/конвертации
  <kind>/<key>/paper.html|paper.pdf|README.md|page.html   скачанный файл
  <kind>/<key>/document.md              Markdown с front matter (результат конвертации)
eval/questions.json                     255 вопросов с позициями ответов в документах
eval/candidates/, eval/selection.json   все кандидаты модели и ручной отбор
eval/REPORT.md, chunking-*.md, hnsw/    результаты оценки retrieval и подбора HNSW
eval/rag/rag-eval.json                  сквозная проверка RAG
```

Если источник недоступен легально, положите PDF вручную в `data/papers/<kind>/<key>/paper.pdf` — следующий
`download` подхватит его (`url: "manual"` в `meta.json`).
