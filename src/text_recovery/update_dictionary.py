import logging
import os
import re
from pathlib import Path

from LoggingSetup import setup_logging


def extract_words_from_text(filepath):
    """Вилучає всі слова з текстового файлу, переводить їх у нижній регістр."""
    filepath = os.path.join('../../data/texts/', filepath)
    logger = logging.getLogger(__name__)
    logger.info(f"Початок вилучення слів з файлу: '{filepath}'")

    words_found = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            logger.debug(f"Розмір завантаженого тексту: {len(text)} символів")

            # Використовуємо регулярний вираз для знаходження всіх послідовностей літер
            found_words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            logger.debug(f"Знайдено {len(found_words)} слів (з повтореннями)")

            words_found.update(found_words)
            logger.info(f"Унікальних слів знайдено: {len(words_found)}")

    except FileNotFoundError:
        logger.error(f"Файл '{filepath}' не знайдено")
        print(f"❌ Помилка: Файл '{filepath}' не знайдено.")
    except Exception as e:
        logger.error(f"Помилка при читанні файлу '{filepath}': {e}")

    return words_found


def load_existing_dictionary(filepath='english_words.txt'):
    """Завантажує слова з наявного файлу словника."""
    logger = logging.getLogger(__name__)
    filepath = os.path.join('../../data/dictionaries/', filepath)
    logger.info(f"Завантаження існуючого словника з: '{filepath}'")

    existing_words = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                word = line.strip().lower()
                if word:
                    existing_words.add(word)

                if line_num % 1000 == 0:
                    logger.debug(f"Оброблено {line_num} рядків словника")

        logger.info(f"Завантажено {len(existing_words)} слів з існуючого словника")

    except FileNotFoundError:
        logger.warning(f"Файл '{filepath}' не знайдено. Буде створено новий")
        print(f"❌ Помилка: Файл '{filepath}' не знайдено. Буде створено новий.")
    except Exception as e:
        logger.error(f"Помилка при завантаженні словника: {e}")

    return existing_words


def save_words_to_file(filepath, words_set):
    """Зберігає множину слів у файл, по одному слову на рядок."""
    logger = logging.getLogger(__name__)
    filepath = os.path.join('../../data/dictionaries/', filepath)
    logger.info(f"Збереження {len(words_set)} слів у файл: '{filepath}'")

    try:
        # Створюємо директорію, якщо вона не існує
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            sorted_words = sorted(list(words_set))

            for i, word in enumerate(sorted_words):
                f.write(word + '\n')

                if (i + 1) % 1000 == 0:
                    logger.debug(f"Записано {i + 1} слів")

        logger.info(f"Успішно збережено словник у '{filepath}'")
        print(f"✅ Оновлений словник збережено у '{filepath}'.")

    except Exception as e:
        logger.error(f"Помилка при збереженні файлу: {e}")
        print(f"❌ Помилка при збереженні файлу: {e}")


def analyze_word_statistics(words_set):
    """Аналізує статистику слів"""
    logger = logging.getLogger(__name__)
    logger.info("Аналіз статистики слів")

    if not words_set:
        logger.warning("Множина слів порожня")
        return

    # Статистика за довжиною
    length_stats = {}
    for word in words_set:
        length = len(word)
        length_stats[length] = length_stats.get(length, 0) + 1

    logger.debug(f"Детальна статистика за довжиною: {dict(sorted(length_stats.items()))}")
    logger.info(f"Діапазон довжини слів: {min(length_stats.keys())}-{max(length_stats.keys())}")

    # Найдовші та найкоротші слова
    min_length = min(len(word) for word in words_set)
    max_length = max(len(word) for word in words_set)

    shortest_words = [word for word in words_set if len(word) == min_length][:10]
    longest_words = [word for word in words_set if len(word) == max_length][:10]

    logger.debug(f"Найкоротші слова ({min_length} символів): {shortest_words}")
    logger.debug(f"Найдовші слова ({max_length} символів): {longest_words}")

    logger.info(
        f"Статистика: мін={min_length}, макс={max_length}, середня={sum(len(w) for w in words_set) / len(words_set):.1f}")


def main():
    """Основна функція скрипту"""
    # Налаштовуємо логування: INFO на консоль, DEBUG у файл
    setup_logging(console_level=logging.INFO, file_level=logging.DEBUG, log_to_file=True)
    logger = logging.getLogger(__name__)

    logger.info("=== Початок оновлення словника ===")
    logger.debug("Детальне логування активовано для файлу")

    alice_text_file = 'alice_in_wonderland.txt'
    dictionary_file = 'english_words.txt'

    # Завантаження слів з тексту Аліси
    logger.info(f"Етап 1: Завантаження слів з '{alice_text_file}'")
    logger.debug(f"Повний шлях до файлу: {os.path.abspath(alice_text_file)}")

    alice_words = extract_words_from_text(alice_text_file)

    if not alice_words:
        logger.error("Не вдалося завантажити слова з тексту Аліси")
        return

    print(f"✅ Знайдено {len(alice_words)} унікальних слів у тексті Аліси.")
    analyze_word_statistics(alice_words)

    # Завантаження існуючого словника
    logger.info(f"Етап 2: Завантаження існуючого словника з '{dictionary_file}'")
    current_dictionary = load_existing_dictionary(dictionary_file)
    print(f"✅ В поточному словнику {len(current_dictionary)} слів.")

    # Пошук унікальних слів
    logger.info("Етап 3: Пошук унікальних слів")
    unique_alice_words = alice_words - current_dictionary
    logger.debug(f"Кількість слів в Alice: {len(alice_words)}")
    logger.debug(f"Кількість слів в існуючому словнику: {len(current_dictionary)}")
    logger.info(f"Знайдено {len(unique_alice_words)} унікальних слів з Аліси")
    print(f"✅ Знайдено {len(unique_alice_words)} унікальних слів з Аліси, яких немає в словнику.")

    if unique_alice_words:
        logger.debug(f"Приклади перших 10 нових слів: {sorted(list(unique_alice_words))[:10]}")

        print("\n✅ Нові слова для додавання:")
        sample_words = sorted(list(unique_alice_words))[:20]
        for word in sample_words:
            print(word)
        if len(unique_alice_words) > 20:
            print("...")

        # Об'єднання словників
        logger.info("Етап 4: Об'єднання словників")
        updated_dictionary = current_dictionary.union(alice_words)
        logger.info(f"Оновлений словник містить {len(updated_dictionary)} слів")
        logger.debug(f"Приріст словника: {len(updated_dictionary) - len(current_dictionary)} слів")
        print(f"\n✅ Оновлений словник міститиме {len(updated_dictionary)} слів.")

        # Збереження оновленого словника
        logger.info("Етап 5: Збереження оновленого словника")
        save_words_to_file(dictionary_file, updated_dictionary)
        print("✅ Словник успішно доповнено лексикою з 'Аліси в Країні Чудес'.")

        # Аналіз оновленого словника
        analyze_word_statistics(updated_dictionary)

    else:
        logger.info("Нових слів для додавання не знайдено")
        logger.debug("Всі слова з Alice вже присутні в існуючому словнику")
        print("\n❌ Не знайдено нових унікальних слів з Аліси для додавання до словника.")

    logger.info("=== Завершення оновлення словника ===")


if __name__ == "__main__":
    main()
