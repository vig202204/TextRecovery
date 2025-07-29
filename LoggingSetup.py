import logging
import os
from datetime import datetime

def setup_logging(console_level=logging.INFO, file_level=logging.DEBUG, log_to_file=True):
    """
    Налаштовує логування для всього проекту з різними рівнями для консолі та файлу
    
    Args:
        console_level: Рівень логування для консолі (за замовчуванням INFO)
        file_level: Рівень логування для файлу (за замовчуванням DEBUG)
        log_to_file: Чи записувати логи у файл
    """
    
    # Створюємо форматери
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Налаштовуємо кореневий логер на найнижчий рівень (щоб усі повідомлення проходили)
    root_logger = logging.getLogger()
    root_logger.setLevel(min(console_level, file_level))
    
    # Очищуємо існуючі handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler з рівнем INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler з рівнем DEBUG (опціонально)
    if log_to_file:
        logs_dir = 'logs'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            print(f"✅ Створено директорію логів: {logs_dir}")
        
        log_filename = f'logs/text_recovery_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        print(f"✅ Файл логів буде створено: {log_filename}")
        print(f"📊 Консоль: {logging.getLevelName(console_level)}, Файл: {logging.getLevelName(file_level)}")
        
        # Тестове повідомлення для перевірки
        test_logger = logging.getLogger('setup_test')
        # test_logger.debug("🔍 DEBUG: Це повідомлення має з'явитися тільки у файлі")
        # test_logger.info("ℹ️ INFO: Це повідомлення з'явиться і в консолі, і у файлі")
        # test_logger.warning("⚠️ WARNING: Це повідомлення з'явиться і в консолі, і у файлі")
    
    return root_logger