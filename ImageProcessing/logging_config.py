"""
Модуль logging_config: настройка логгера приложения через YAML-конфиг.

- Конфиг логгера лежит в отдельном файле logging.yml
- Поддерживаются:
  - лог в файл (DEBUG, подробно)
  - лог в консоль (INFO, кратко)
- Опционально можно переопределить путь/имя лог-файла из кода
"""

import logging
import logging.config
import os
from typing import Optional

import yaml  # не забудь установить: pip install pyyaml


def setup_logging(
    config_path: str = "logging.yml", # путь к YAML-файлу с конфигом
    log_dir: Optional[str] = None, # опциональная директория для файла логов
    log_file: Optional[str] = None, # опциональное имя файла логов
    default_level: int = logging.INFO, # уровень логирования по умолчанию, если YAML не нашли
) -> logging.Logger:
    

    # Проверка на существование YAML-конфига
    if os.path.exists(config_path): # проверяем: существует ли файл logging.yml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) # превращает YAML в обычный dict

        # Переопределяем файл логов, если надо
        file_handler_cfg = config.get("handlers", {}).get("file") # достаём хендлер с ключом "file" из YAML
        if file_handler_cfg is not None: # Если он есть, то можем изменить его настройки
            # базовое имя из YAML
            filename = file_handler_cfg.get("filename", "app.log")

            # если передали имя файла — подменяем
            if log_file:
                filename = log_file

            # если передали директорию — собираем полный путь
            if log_dir:
                os.makedirs(log_dir, exist_ok=True) # берём только имя файла
                filename = os.path.join(log_dir, os.path.basename(filename))
               
            else:
                # если в YAML был путь типа "logs/app.log" — убеждаемся, что папка есть
                dirname = os.path.dirname(filename)
                if dirname:
                    os.makedirs(dirname, exist_ok=True) # гарантируем. что папка существует

            file_handler_cfg["filename"] = filename # обновляем конфиг хендлера file

        logging.config.dictConfig(config) # Применяем конфигурацию логирования
    #если yaml нету    
    else:
        # fallback, если logging.yml не найден
        logging.basicConfig(level=default_level) # настраиваем простейший логгер
        logging.getLogger(__name__).warning( #берём логгер текущего модуля и пишем Warning
            "Файл конфигурации логгера %s не найден, используется basicConfig",
            config_path,
        )

    return logging.getLogger("cat_app") # возвращаем готовый логгер с именем cat_app


