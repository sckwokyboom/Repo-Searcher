# Repo-Searcher Frontend

SPA-приложение для поиска по Java-репозиториям. Построено на React 19 + TypeScript + Vite + Material UI 7.

## Требования

- Node.js 18+
- npm 9+

## Установка и запуск

```bash
npm install
npm run dev
```

Dev-сервер запустится на http://localhost:5173

## Переменные окружения

| Переменная     | По умолчанию            | Описание          |
|----------------|-------------------------|--------------------|
| `VITE_API_URL` | `http://localhost:8000` | URL бэкенд-сервера |

## Скрипты

| Команда           | Описание                          |
|-------------------|-----------------------------------|
| `npm run dev`     | Запуск dev-сервера (Vite)         |
| `npm run build`   | Сборка для продакшена             |
| `npm run preview` | Предпросмотр собранного приложения |
| `npm run lint`    | Проверка кода через ESLint        |

## Структура проекта

```
src/
├── pages/              # Страницы приложения
│   ├── HomePage.tsx    #   Поиск и индексация репозиториев
│   └── RepoPage.tsx    #   Поиск кода и результаты
├── components/         # UI-компоненты
│   ├── Layout.tsx      #   Общий layout
│   ├── RepoSearchBar.tsx   # Поиск репозиториев на GitHub
│   ├── RepoCard.tsx        # Карточка репозитория
│   ├── CodeSearchBar.tsx   # Поиск по коду
│   ├── SearchResultList.tsx # Список результатов
│   ├── SearchResultCard.tsx # Карточка результата
│   ├── CodeBlock.tsx       # Подсветка синтаксиса
│   ├── CallGraphPanel.tsx  # Граф вызовов (force-graph)
│   └── IndexingProgress.tsx # Прогресс индексации
├── services/           # API-клиенты
│   ├── api.ts          #   Axios-инстанс
│   ├── repoService.ts  #   Работа с репозиториями
│   ├── searchService.ts #  Поиск по коду
│   └── graphService.ts #   Граф вызовов
├── hooks/              # Кастомные React-хуки
├── types/              # TypeScript-типы
├── theme.ts            # Тёмная тема MUI (purple/cyan)
├── App.tsx             # Роутинг
└── main.tsx            # Точка входа
```

## Основные зависимости

- **React 19** + **React Router 7** — UI и роутинг
- **MUI 7** — компоненты Material Design
- **Axios** — HTTP-клиент
- **prism-react-renderer** — подсветка синтаксиса
- **react-force-graph-2d** — визуализация графа вызовов

## Взаимодействие с бэкендом

- REST API через Axios (base URL настраивается через `VITE_API_URL`)
- WebSocket для отслеживания прогресса индексации (`ws://localhost:8000/api/ws/indexing/{repo_id}`)
