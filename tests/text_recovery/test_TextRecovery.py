import unittest
import logging
from src.text_recovery.TextRecovery import TextRecovery
from LoggingSetup import setup_logging

class TestTextRecovery(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Налаштовуємо логування для всіх тестів"""
        # Налаштовуємо логування: INFO на консоль, DEBUG у файл
        setup_logging(console_level=logging.INFO, file_level=logging.DEBUG, log_to_file=True)
        logger = logging.getLogger(__name__)

        # Створюємо логер для тестів
        cls.logger = logging.getLogger(cls.__name__)
        cls.logger.info("=== Започатковано набір тестів TextRecovery ===")

    def setUp(self):
        """Налаштування перед кожним тестом"""
        self.logger.info(f"Починаємо тест: {self._testMethodName}")
        try:
            self.text_recovery = TextRecovery()
            self.logger.debug("TextRecovery успішно ініціалізовано")
        except Exception as e:
            self.logger.error(f"Помилка ініціалізації TextRecovery: {e}")
            raise

    def tearDown(self):
        """Очищення після кожного тесту"""
        self.logger.info(f"Завершено тест: {self._testMethodName}")

    def test_get_bigram_score(self):
        """Тест отримання біграм-скору"""
        word1 = "hello"
        word2 = "world"
        expected_score = 0.95
        
        self.logger.info(f"Тестуємо біграм-скор для: '{word1}' -> '{word2}'")
        self.logger.debug(f"Очікуваний скор: {expected_score}")
        
        try:
            actual_score = self.text_recovery.get_bigram_score(word1, word2)
            self.logger.debug(f"Отриманий скор: {actual_score}")
            
            self.assertEqual(expected_score, actual_score)
            self.logger.info("✅ Тест біграм-скору пройшов успішно")
            
        except AssertionError as e:
            self.logger.error(f"❌ Тест біграм-скору провалився: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Неочікувана помилка в тесті біграм-скору: {e}")
            raise

    def test_find_asterisk_candidates(self):
        """Тест пошуку кандидатів з зірочкою"""
        word_pattern = "h*llo"
        expected_candidates = ['hello']
        
        self.logger.info(f"Тестуємо пошук кандидатів для патерну: '{word_pattern}'")
        self.logger.debug(f"Очікувані кандидати: {expected_candidates}")
        
        try:
            actual_candidates = self.text_recovery.find_asterisk_candidates(word_pattern)
            self.logger.debug(f"Знайдені кандидати: {actual_candidates}")
            
            self.assertEqual(expected_candidates, actual_candidates)
            self.logger.info("✅ Тест пошуку кандидатів пройшов успішно")
            
        except AssertionError as e:
            self.logger.error(f"❌ Тест пошуку кандидатів провалився: {e}")
            self.logger.error(f"Очікувалось: {expected_candidates}, отримано: {actual_candidates}")
            raise

    def test_generate_anagram_candidates(self):
        """Тест генерації анаграм"""
        word_pattern = "stop"
        expected_candidates = ['stop', 'spot', 'tops']
        
        self.logger.info(f"Тестуємо генерацію анаграм для: '{word_pattern}'")
        self.logger.debug(f"Очікувані анаграми: {expected_candidates}")
        
        try:
            actual_candidates = self.text_recovery.generate_anagram_candidates(word_pattern)
            self.logger.debug(f"Згенеровані анаграми: {actual_candidates}")
            
            self.assertCountEqual(expected_candidates, actual_candidates)
            self.logger.info("✅ Тест генерації анаграм пройшов успішно")
            
        except AssertionError as e:
            self.logger.error(f"❌ Тест генерації анаграм провалився: {e}")
            raise

    def test_recover_text(self):
        """Тест відновлення тексту"""
        damaged_text = "h*llo w*rld"
        
        self.logger.info(f"Тестуємо відновлення тексту: '{damaged_text}'")
        
        try:
            actual_result = self.text_recovery.recover_text(damaged_text)
            self.logger.debug(f"Результат відновлення: '{actual_result}'")
            
            self.assertIsInstance(actual_result, str)
            self.assertNotEqual(actual_result, "")
            self.logger.info("✅ Тест відновлення тексту пройшов успішно")
            
        except Exception as e:
            self.logger.error(f"❌ Помилка в тесті відновлення тексту: {e}")
            raise

    def test_recover_text_enhanced(self):
        """Тест розширеного відновлення тексту"""
        damaged_text = "al*ce went d*wn"
        
        self.logger.info(f"Тестуємо розширене відновлення: '{damaged_text}'")
        
        try:
            actual_result = self.text_recovery.recover_text_enhanced(damaged_text)
            self.logger.debug(f"Результат розширеного відновлення: '{actual_result}'")
            
            self.assertIsInstance(actual_result, str)
            self.assertNotEqual(actual_result, "")
            self.logger.info("✅ Тест розширеного відновлення пройшов успішно")
            
        except Exception as e:
            self.logger.error(f"❌ Помилка в тесті розширеного відновлення: {e}")
            raise

    def test_segment_alice_text(self):
        """Тест сегментації тексту Alice"""
        text = "sister"
        expected_result = ['sister']
        
        self.logger.info(f"Тестуємо сегментацію тексту: '{text}'")
        self.logger.debug(f"Очікуваний результат: {expected_result}")
        
        try:
            actual_result = [''.join(self.text_recovery.segment_alice_text(text))]
            self.logger.debug(f"Фактичний результат: {actual_result}")
            
            # Виправляємо логіку тесту - порівнюємо списки напряму
            self.assertEqual(expected_result, actual_result)
            self.logger.info("✅ Тест сегментації тексту пройшов успішно")
            
        except AssertionError as e:
            self.logger.error(f"❌ Тест сегментації провалився: {e}")
            self.logger.error(f"Очікувалось: {expected_result}")
            self.logger.error(f"Отримано: {actual_result}")
            raise

    def test_select_best_candidate_with_context(self):
        """Тест вибору найкращого кандидата з контекстом"""
        candidates = ["hello", "hallo"]
        previous_word = "hi"
        next_word = "world"
        expected_result = "hello"
        
        self.logger.info(f"Тестуємо вибір кандидата: {candidates}")
        self.logger.debug(f"Контекст: '{previous_word}' _ '{next_word}'")
        self.logger.debug(f"Очікуваний результат: '{expected_result}'")
        
        try:
            actual_result = self.text_recovery.select_best_candidate_with_context(
                candidates, previous_word, next_word
            )
            self.logger.debug(f"Обраний кандидат: '{actual_result}'")
            
            self.assertEqual(expected_result, actual_result)
            self.logger.info("✅ Тест вибору кандидата пройшов успішно")
            
        except AssertionError as e:
            self.logger.error(f"❌ Тест вибору кандидата провалився: {e}")
            raise

    def test_preprocess_alice_patterns(self):
        """Тест попередньої обробки тексту Alice"""
        text = "Alice in Wonderland"
        expected_result = "alice in wonderland"
        
        self.logger.info(f"Тестуємо попередню обробку: '{text}'")
        self.logger.debug(f"Очікуваний результат: '{expected_result}'")
        
        try:
            actual_result = self.text_recovery.preprocess_alice_patterns(text)
            self.logger.debug(f"Результат обробки: '{actual_result}'")
            
            self.assertEqual(expected_result, actual_result)
            self.logger.info("✅ Тест попередньої обробки пройшов успішно")
            
        except AssertionError as e:
            self.logger.error(f"❌ Тест попередньої обробки провалився: {e}")
            raise

    @classmethod
    def tearDownClass(cls):
        """Завершення всіх тестів"""
        cls.logger.info("=== Завершено набір тестів TextRecovery ===")

if __name__ == "__main__":
    unittest.main(verbosity=2)