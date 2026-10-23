"""Лаба 1: прогоны LLM через OpenAI-совместимый REST Ollama и их оценка.

  config      — пути, config.yaml и gold.json, сборка сообщений и параметров
  llm_client  — запрос к /v1/chat/completions со стримингом и замерами, замер prefill
  runner      — основной прогон и разовый эксперимент (trial)
  scoring     — оценка ответа под задачу по эталонам
  analysis    — метрики, сводные таблицы, answers.md
  plots       — графики
"""
