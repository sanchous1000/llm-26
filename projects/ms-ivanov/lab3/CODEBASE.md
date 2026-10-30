# Устройство кода

Справочник для разработчика: из чего состоит проект, как связаны части, какие есть команды, конфиги и данные.
Отчёт по лабораторной и результаты — в [README.md](README.md). Все команды запускаются из каталога `lab3/`:
`.venv/bin/python -m source <команда> [--help]` (Python 3.14).

## 1. Общая схема

```text
 корпус (Лаба 1)          индекс (Лаба 2)                 сервис (Лаба 2)              агенты (Лаба 3)
 ───────────────          ───────────────                 ───────────────              ───────────────
 data/sources.bib   download   data/papers/…/document.md   chunk → embed    Qdrant +     MCP-сервер papers-rag
 data/SOURCES.md  ─────────►  (Markdown + front matter) ──────────────────► PostgreSQL ─► (streamable HTTP, nginx,
 rag_additions*   convert                                  BGE-M3 hybrid                 https://thesis.rag.mooo.com/mcp)
                                                                                              │ 4 инструмента поиска
                                                     ┌────────────────────────────────────────┤
                                                     ▼                                        ▼
                                   свой агент (PydanticAI)                    OpenCode CLI ──► LLM-прокси
                                   source/agent/own/                          source/agent/opencode/
                                             │                                        │
                                             └──────────── Langfuse (локально) ◄──────┘
                                                           сессии, трейсы, датасет, эксперименты, оценки
```

- **Тяжёлая работа** (скачивание, разбор, эмбеддинги на Apple MPS) выполняется локально. На сервере (3 vCPU, без GPU)
  работает только поиск: MCP-сервер с индексом, nginx и мониторинг. Индекс переносится скриптом `deploy/deploy.sh`.
- **Модель агентов** — `deepseek-v4.1-flash` через OpenAI-совместимый эндпоинт opencode
  (`https://opencode.ai/inference/go/openai/v1`, обязателен заголовок `x-opencode-session`). Судья — `glm-5.3`
  у того же провайдера.

## 2. Структура репозитория

```text
lab3/
├── README.md               отчёт по лабораторной (формат — projects/README.md)
├── CODEBASE.md             этот файл
├── source/                 Python-пакет `source` (CLI: python -m source) и тесты
├── configs/
│   ├── default.yaml        индексация и RAG: чанкинг, эмбеддинги, HNSW, параметры поиска
│   └── agent/              агенты: agent.yaml (свой), opencode.yaml (OpenCode), judge.yaml (судья) и промпты
├── data/                   списки источников (*.bib, *.md); корпус data/papers/ в git не хранится
├── eval/
│   ├── questions.json      255 вопросов с цитатами-доказательствами (оценка поиска, index-check, rag-eval)
│   └── agent/              набор задач агентов, результаты прогонов, таблицы, скриншоты Langfuse
├── deploy/
│   ├── deploy.sh           выкладка на сервер;  tokens.sh — токены MCP (запускается на сервере)
│   ├── agent-prompt.md     инструкция для внешних агентов, которые подключают papers-rag
│   ├── langfuse/           docker-compose локального Langfuse и генератор его .env
│   ├── nginx/              шаблоны конфигурации и выбор http/https при старте
│   └── monitoring/         Prometheus, разбор логов nginx, Grafana (дашборд генерируется make_dashboard.py)
├── Dockerfile              образ MCP-сервиса (python:3.14-slim, torch CPU)
├── docker-compose.yml      Qdrant + PostgreSQL; профиль app — mcp, nginx, certbot, prometheus, grafana, экспортёры
├── pyproject.toml          black, ruff, pytest
├── .env.example            шаблон переменных окружения (сам .env в git не попадает)
└── archive/                старые документы и разовые результаты (в .gitignore, только локально)
```

## 3. Пакет `source`

Слои зависят только от нижележащих: `agent` → (MCP по сети) → `rag` → `indexing` → `parsing`/`storage`.
Агентный слой не импортирует `rag`: с индексом он работает только через MCP-сервер, как любой внешний агент.

### Корпус: сбор и разбор

| модуль | назначение |
|---|---|
| `cli.py`, `__main__.py` | точка входа; команды `download`, `convert`, `report`, регистрация команд остальных слоёв |
| `config.py` | пути по умолчанию (`data/`, `eval/`), паузы и повторы сети, имена файлов |
| `bibliography/` | `sources.bib` + каталог `SOURCES.md` (приоритет ★, теги) + дополнения `rag_additions*.bib` → `Source` |
| `net/` | HTTP-клиент: паузы по хостам, повторы, атомарная запись, отсев страниц-заглушек с кодом 200 |
| `downloaders/` | загрузчики по типу источника: arXiv (HTML, затем PDF), DOI (открытые PDF через OpenAlex и Semantic Scholar), GitHub README, веб |
| `storage/` | состояние каждой записи (`meta.json`) и общий `manifest.json` |
| `pipeline.py`, `report.py` | возобновляемый цикл загрузки и сводка по корпусу |
| `parsing/`, `conversion_pipeline.py` | файлы → Markdown с YAML front matter: `latexml/` (arXiv HTML), `pdf`, `readme`, `webpage`; параллельно и возобновляемо |
| `parsing/document_body.py` | текст документа без front matter, границы раздела литературы |

### Индекс

| модуль | назначение |
|---|---|
| `indexing/config.py` | конфигурация пайплайна (`configs/default.yaml`, `--set ключ=значение`); хэши чанкинга и эмбеддинга дают имя коллекции |
| `indexing/blocks.py` → `units.py` → `splitters/` → `chunking.py` | Markdown → блоки → неделимые единицы (формула с пояснением, таблица с подписью) → чанки |
| `indexing/embedding/` | BGE-M3: dense и sparse векторы за один проход; fp16 на MPS, fp32 на CPU |
| `indexing/db/` | SQLAlchemy-схема PostgreSQL: документы, наборы чанков, чанки, коллекции |
| `indexing/vectorstore.py`, `search.py` | коллекция Qdrant (named vectors `dense`/`sparse`, HNSW) и поиск dense/sparse/hybrid (RRF) |
| `indexing/pipeline.py` | `chunk` → PostgreSQL, `embed` → Qdrant; обе стадии идемпотентны |
| `indexing/retrieval_check.py` | Recall@k / MRR коллекции по `eval/questions.json` |
| `indexing/commands.py` | команды `db-init`, `chunk`, `embed`, `index`, `index-status`, `index-drop`, `index-check`, `index-apply` |

### RAG и MCP-сервер

| модуль | назначение |
|---|---|
| `rag/retriever.py`, `service.py` | запрос → ранжированные фрагменты с метаданными для цитирования |
| `rag/prompt.py`, `locate.py` | сборка промпта и разбор ответа с цитатами; поиск цитаты в тексте (точно, затем нечётко) |
| `rag/engine.py`, `llm.py`, `opencode.py` | вопрос → фрагменты → модель (`opencode run`) → ответ с источниками |
| `rag/mcp_server.py`, `metrics.py` | MCP-инструменты `search_papers`, `get_chunk_context`, `list_documents`, `get_document_outline`; метрики Prometheus |
| `rag/evaluation.py` | сквозная проверка RAG с LLM-судьёй (`rag-eval`) |
| `rag/commands.py`, `render.py` | команды `ask`, `chat`, `search`, `mcp`, `mcp-config`, `rag-eval` и вывод в терминал |

### Агенты (`source/agent/`)

Три подпакета по ролям и два общих модуля. Все подпакеты используют `config` и `tracing`; `evaluation` запускает
`own` и `opencode`; `opencode` берёт у `own` только калькулятор (`own/tools.py`), обратных зависимостей нет.

```text
source/agent/
├── config.py, tracing.py   общие: схема конфигов и Langfuse
├── commands.py             CLI агентного слоя
├── own/                    свой агент на PydanticAI
├── opencode/               готовый агент OpenCode и LLM-прокси
└── evaluation/             набор задач, эксперименты, судья, метрики, таблицы
```

| модуль | назначение |
|---|---|
| `config.py` | pydantic-схема `agent.yaml`/`opencode.yaml`, переопределения `--set`, загрузка промптов и `.env` |
| `tracing.py` | обёртка Langfuse: без ключей все вызовы пустые; сообщения в читаемом виде, стоимость, уровни статусов |
| `commands.py` | CLI агентного слоя (см. раздел 5) |
| `own/mcp_client.py` | клиент papers-rag: короткая сессия на каждый вызов, повторы при сетевых сбоях, не больше 3 вызовов одновременно |
| `own/tools.py` | инструменты агента: 4 обёртки MCP (+ span `retrieval` с чанками) и безопасный калькулятор по AST; счётчики реплики `TurnState` |
| `own/context.py` | подсчёт токенов (tiktoken `o200k_base`) и сокращение истории: стратегии `summary`, `window`, `none` |
| `own/runtime.py` | capability PydanticAI: перед каждым запросом к модели — контроль контекста, затем логирование генераций и вызовов инструментов |
| `own/agent.py` | `AgentFactory` (общее для конфигурации) и `DialogSession.ask` — одна реплика со всеми остановами и восстановлением |
| `own/runner.py` | сборка: конфиг → MCP → сессия; прогон диалога целиком |
| `opencode/runner.py` | готовый агент: `opencode.json` диалога, `opencode run --session`, разбор событий, запись реплики |
| `opencode/proxy.py`, `proxy_stream.py` | OpenAI-совместимый LLM-прокси для OpenCode: пропускает SSE-поток, пишет генерации и инструменты в Langfuse |
| `opencode/proxy_server.py` | запуск прокси: отдельно (`llm-proxy`), в текущем event loop или в фоновом потоке |
| `opencode/calc_mcp.py` | калькулятор своего агента как stdio MCP-сервер для OpenCode |
| `evaluation/dataset.py` | набор задач `eval/agent/tasks.yaml`: загрузка, проверка схемы, ключевые факты, экспорт в формат Langfuse |
| `evaluation/run.py` | датасет и эксперименты Langfuse; файлы прогонов в `eval/agent/runs/` |
| `evaluation/judge.py` | оценка реплик (точная проверка, правило или LLM-судья) и запись оценок в Langfuse |
| `evaluation/metrics.py`, `report.py` | метрики по репликам, задачам и прогонам; таблицы `eval/agent/RESULTS.md` и `RESULTS.json` |

## 4. Агентный слой подробно

### Реплика своего агента

```text
DialogSession.ask(text)                     trace "turn N" (agent): session = диалог, теги agent/config/model/context
 └─ _run_guarded: таймаут turn_timeout_s, UsageLimits(max_steps, max_tool_calls)
     └─ Agent.iter: итерация N (chain)
         1. AgentRuntime.before_model_request → ContextManager.process (span context-reduction → reduce:*)
         2. запрос к модели (generation llm: промпт, параметры, usage, стоимость)
         3. вызовы инструментов (tool:*; у search_papers вложенный retriever)
     ответ без вызовов инструментов → конец реплики
```

| ситуация | что происходит | уровень в Langfuse |
|---|---|---|
| ответ без вызовов инструментов | конец реплики, статус `ok` | DEFAULT |
| ошибка инструмента | текст ошибки уходит модели как результат, до `tool_retries` повторов | ERROR у span инструмента |
| исчерпаны `max_steps` или `max_tool_calls` | `step_limit`: помощник пишет ответ из собранного в реплике материала | WARNING |
| `turn_timeout_s` | `timeout`: сообщение о нехватке времени | WARNING |
| любая другая ошибка | `error`: принудительный ответ или сообщение об ошибке | ERROR |

Диалог продолжается после любого исхода. История сохраняется без «висящих» вызовов инструментов.

### Управление контекстом (`own/context.py`)

`limit_tokens` (32 000) — бюджет всего промпта: системный промпт, описания инструментов и история. Размер считается
перед каждым запросом к модели. Реплика — это сообщение пользователя и всё до следующего, и сокращение её не
разрывает. Шаги, пока промпт больше лимита:

1. результаты инструментов и reasoning прошлых реплик обрезаются до `old_tool_result_chars` (300);
2. стратегия: `summary` — реплики старше `keep_recent_turns` заменяются одним резюме (`configs/agent/summary_prompt.md`;
   при сбое резюме откат к `window`), `window` — старые реплики отбрасываются, `none` — только измерение;
3. в текущей реплике обрезаются старые результаты, два последних пакета результатов остаются целыми.

Почему лимит 32k, а не 16k, объяснено в [eval/agent/context-limit.md](eval/agent/context-limit.md).

### OpenCode через прокси

```text
OpencodeSession.ask ── trace "turn N" открывается здесь; его trace id и id наблюдения → заголовки x-lab3-*
  └─ opencode run --standalone --auto --format json --agent papers --session ses_… (env PWD = рабочий каталог)
       opencode.json: провайдер lab3 → http://127.0.0.1:4100/v1, MCP papers-rag + calc, steps = max_steps,
       limit.context = limit_tokens + max_output_tokens, compaction buffer 0, встроенные инструменты запрещены
         └─ LLM-прокси: добавляет ключ и x-opencode-session, пропускает поток, логирует generation и tool:*,
            суммирует usage по trace (GET /usage/<trace_id>: события OpenCode теряют usage части шагов)
```

- OpenCode 2 вызывает MCP-инструменты из кода в инструменте `execute`. Реальные вызовы прокси и раннер извлекают из
  этого кода (`proxy_stream.code_tool_calls`).
- Сырые события каждой реплики сохраняются в её рабочий каталог (`turn-NN.events.jsonl`, временный каталог `opencode-*`).
- Записи реплик обоих агентов имеют один формат (`TurnResult.to_dict` и `OpencodeSession._record`), поэтому метрики и
  судья у них общие.

### Оценка

```text
agent-eval-upload   tasks.yaml → датасет papers-agent-tasks (id элемента = id задачи)
agent-eval-run      эксперимент Langfuse: элемент = диалог → eval/agent/runs/<run>.jsonl, <run>.meta.json
agent-eval-judge    вердикт на каждую реплику → <run>.judged.jsonl; оценки turn_correct, task_success, success_rate…
agent-eval-report   judged-файлы → eval/agent/RESULTS.md + RESULTS.json
```

- Способы оценки реплики: `exact` — только ключевые факты (`check: exact` в задаче); `rule` — пустой или служебный
  ответ сразу неверен; `llm` — судья по эталону и фактам (`configs/agent/judge_prompt.md`).
- При повторной оценке хорошие вердикты из прежнего judged-файла переиспользуются, переоцениваются только сбои судьи.
  Id оценок в Langfuse детерминированы (uuid5), поэтому повторная отправка перезаписывает их.
- Judged-файл содержит те же записи, что и сырой, плюс вердикты. Поэтому в git хранится только он, а
  `load_records` читает его, если сырого файла нет.

## 5. Команды

| группа | команды |
|---|---|
| корпус | `download` (`--kinds`, `--keys`, `--limit`, `--pdf`, `--force`, `--retry-missing`), `convert` (`--workers`, `--force`, `--no-references`), `report` |
| индекс | `db-init`, `chunk`, `embed`, `index` (`--rebuild`), `index-status`, `index-drop`, `index-check` (`--vary`, `--lang`, `--markdown`), `index-apply` |
| RAG | `ask`, `chat`, `search`, `mcp` (`--transport`, `--metrics-port`), `mcp-config`, `rag-eval` |
| свой агент | `agent-tasks` (`--export`), `agent-ask "q1" "q2"`, `agent-chat` (`/ctx`, `/reset`, `/exit`), `agent-dialog --task D07` |
| OpenCode | `llm-proxy`, `opencode-ask`, `opencode-dialog --task D07` |
| оценка | `agent-eval-upload`, `agent-eval-run --agent own\|opencode --run-name …`, `agent-eval-judge --run-name …`, `agent-eval-report --runs …` |

У агентных команд есть `--config` и `--set ключ=значение`. Пример: `--set context.strategy=window --set context.limit_tokens=8000`.

## 6. Конфигурация и секреты

| файл | что задаёт |
|---|---|
| `configs/default.yaml` | чанкинг (structured, 1024 токена BGE-M3), эмбеддинги (hybrid), HNSW (`m` 16, `ef` 64), `min_priority`, параметры RAG |
| `configs/agent/agent.yaml` | свой агент: модель и цены, MCP (`token_env`), лимиты (12 шагов, 24 вызова, 300 с), контекст (summary, 32k) |
| `configs/agent/opencode.yaml` | те же поля для OpenCode + `opencode` (агент, провайдер, запрещённые инструменты) и `proxy` (порт 4100) |
| `configs/agent/judge.yaml` | модель судьи, цены, параллельность |
| `configs/agent/*.md`, `tools.yaml` | системный промпт, шаблон резюме, промпт судьи, описания инструментов для модели |
| `.env` (шаблон `.env.example`) | БД, порты, домен, `OPENCODE_API_KEY`, `PAPERS_RAG_TOKEN_AGENT`/`_OPENCODE`, `LANGFUSE_*` |
| `deploy/langfuse/.env` | секреты локального Langfuse; создаётся `deploy/langfuse/init-env.sh` |

Имя коллекции Qdrant — это хэши конфигураций чанкинга и эмбеддинга (рабочая коллекция — `chunks_04e74091_b0b9126a`).
В хэш эмбеддинга входят `device` и `dtype`, поэтому менять их в `default.yaml` нельзя: имя разойдётся с загруженной
коллекцией.

## 7. Данные: что в git, что генерируется

| путь | в git | откуда |
|---|---|---|
| `data/*.bib`, `data/*.md` | да | списки источников (генерируются во внешнем репозитории исследования) |
| `data/papers/` | нет | `download` + `convert` (≈ 1 ГБ) |
| `eval/questions.json` | да | набор вопросов Лабы 2 (код генерации остался в ветке `lab2`) |
| `eval/rag/` | нет | `rag-eval` |
| `eval/agent/tasks.yaml` | да | набор задач агентов (вручную) |
| `eval/agent/tasks.jsonl` | нет | `agent-tasks --export` |
| `eval/agent/runs/*.judged.jsonl`, `*.meta.json` | да | `agent-eval-judge`, `agent-eval-run` |
| `eval/agent/runs/*.jsonl` (сырые) | нет | `agent-eval-run` |
| `eval/agent/RESULTS.md`, `RESULTS.json` | да | `agent-eval-report` |
| `archive/` | нет | старые планы, контекст Лабы 2, отчёты Лабы 2 о подборе чанкинга и HNSW, пробные прогоны |

## 8. Инфраструктура

**Локально.**
- `docker compose up -d` поднимает Qdrant и PostgreSQL для индексации.
- Langfuse запускается командой `cd deploy/langfuse && ./init-env.sh && docker compose up -d`
  (http://localhost:3000). В нём web, worker, Postgres, ClickHouse и Redis, а S3-хранилище заменено на RustFS.
  Langfuse 4 работает в режиме events_only, поэтому данные читаются через `/observations`.

**Сервер** (`thesis.rag.mooo.com`, compose-проект с профилем `app`).
- nginx работает в сети хоста. Он проверяет токен (`Authorization: Bearer`) и ограничивает частоту до 10 запросов/с на IP.
- Остальные сервисы: MCP-сервер, Qdrant и PostgreSQL, а также Prometheus, Grafana (`/grafana/`) и экспортёры.

Операции:
- `deploy/deploy.sh <host> [--code-only]`.
  - Без флага: синхронизирует код и заменяет серверный индекс локальным (`pg_dump`, снимок коллекции Qdrant).
  - С `--code-only`: только код и конфиги. Синхронизация идёт с `--delete`, поэтому правки на сервере затираются:
    правьте шаблоны в репозитории.
- Токены MCP выдаются только по SSH на сервере:
  - `./tokens.sh create <имя>`, `list`, `revoke <имя>`;
  - у агентов свои токены: `lab3-agent` и `lab3-opencode`.
- Дашборд Grafana описан кодом в `deploy/monitoring/make_dashboard.py`. После правок перегенерируйте
  `grafana/dashboards/rag-system.json`.
- На сервере есть чужие проекты (`uni-gpu-mcp` и другие), их не трогать. Не запускайте `docker system prune`.

Подводные камни, на которые уже наступали:
- **arXiv и ar5iv:** ar5iv отвечает 200 на страницы-заглушки, а arxiv.org требует паузу 15 с между запросами.
- **AMS** отвечает 429 на скрипты.
- **Внешний DNS в песочнице:** домен может не резолвиться. Помогает
  `curl --resolve thesis.rag.mooo.com:443:<ip> --noproxy '*'`.
- **Hugging Face:** через прокси помогает `HF_HUB_DISABLE_XET=1`.
- **OpenCode** доверяет `PWD`, а не рабочему каталогу процесса. Id сессии должен начинаться с `ses`.

## 9. Тесты и стиль

```bash
.venv/bin/python -m pytest -q                         # 279 тестов, без сети (Qdrant — локальный режим)
.venv/bin/python -m pytest --cov=source -q            # покрытие
.venv/bin/black --check source deploy && .venv/bin/ruff check source deploy
```

- Тесты лежат в `source/tests/`, имя файла соответствует слою (`test_agent_*`, `test_indexing_*`, `test_rag*` …).
- Сеть и модели в тестах подменяются: `FunctionModel` PydanticAI вместо LLM, фейковые MCP-бэкенд и клиент Langfuse,
  `httpx.MockTransport` для прокси.
- Стиль: PEP 8, black/ruff (`pyproject.toml`), типы в сигнатурах, неизменяемые dataclass-ы для данных, короткие функции.
  Тяжёлые библиотеки (torch, transformers, агентный стек в CLI) импортируются лениво.

Как расширять:
- **Новый инструмент агента:** функция в `source/agent/own/tools.py`, описание в `configs/agent/tools.yaml`, тест в
  `test_agent_wiring.py`.
- **Новая команда:** функция `run_*` и парсер в `commands.py` своего слоя, затем запись в `HANDLERS`.
- **Новая задача:** запись в `eval/agent/tasks.yaml`, затем проверка через `agent-tasks`.
